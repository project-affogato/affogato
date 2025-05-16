import json
import os
import time
from typing import Literal, Optional

import numpy as np
import redis
from PIL import Image

from src.dataset.objaverse.dataset import ObjaverseDatasetTar
from src.job_manager import ack_job, fetch_next_job, generate_unique_id
from src.utils.logging import get_logger
from src.utils.timer import Timer

logger = get_logger(__name__)


def instantiate_molmo(backend: Literal["lmdeploy", "hf", "vllm"] = "lmdeploy"):
    if backend == "hf":
        from src.model.molmo import Molmo as MolmoHF

        return MolmoHF(model_id="allenai/Molmo-7B-D-0924", use_tqdm=False)
    elif backend == "vllm":
        from src.model.vllm.molmo import Molmo as MolmoVLLM

        return MolmoVLLM(model_id="allenai/Molmo-7B-D-0924", use_tqdm=False)
    else:
        raise ValueError(f"Invalid backend: {backend}")


def run_stage2_molmo(
    uid: str,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    backend: Literal["lmdeploy", "hf", "vllm"] = "lmdeploy",
    overwrite: bool = False,
    redis_host: str = "141.223.85.85",
    redis_port: int = 6379,
):
    logger.info("==> Running stage 2: molmo")

    redis_conn = redis.Redis(host=redis_host, port=redis_port, db=0)
    logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

    # Fetch hyperparameters from Redis
    config_json = redis_conn.hget(f"experiment:{uid}", "config")
    if not config_json:
        raise ValueError(f"No hyperparameters found for {uid}")

    hparams = json.loads(config_json)
    logger.info(f"Fetched hyperparameters: {hparams}")

    required_keys = [
        "image_skip",
        "num_images",
        "subset",
        "filename",
        "query_filename",
    ]
    assert all(key in hparams for key in required_keys), (
        f"Missing required keys: {required_keys}"
    )
    total_count = int(redis_conn.hget(f"experiment:{uid}", "total_count"))

    dataset = ObjaverseDatasetTar(
        data_dir=data_dir or hparams["data_dir"],
        subset=hparams["subset"],
        load_image_only=True,
        render_dir="gobjaverse_reduced_images",
        num_images=int(hparams["num_images"]),
        image_skip=int(hparams["image_skip"]),
    )
    molmo = instantiate_molmo(backend)

    # Generate a unique consumer ID for this worker
    consumer_id = f"worker-{generate_unique_id()}"
    logger.info(f"Worker ID: {consumer_id}, Job ID: {uid}")

    save_dir = save_dir or hparams["save_dir"]
    filename = hparams["filename"]
    query_filename = hparams["query_filename"]
    timer = Timer()
    while True:
        job_id, job = fetch_next_job(redis_conn, uid, consumer_id)
        if job is None:
            logger.info("No job found, sleeping for 1 second")
            time.sleep(1)
            continue

        try:
            sample = dataset[int(job["idx"])]
        except Exception as e:
            logger.warning(
                f"Error processing {dataset.object_ids[int(job['idx'])]}: {e}"
            )
            continue
        images = sample["images"]
        image_wh = sample["image_wh"]
        object_id = sample["object_id"]
        instance_dir = os.path.join(save_dir, object_id)
        queries_path = os.path.join(instance_dir, query_filename)
        points_path = os.path.join(instance_dir, filename)

        # Skip if file already exists
        if os.path.exists(points_path) and not overwrite:
            logger.warning(f"Skipping {object_id} because it already exists")
            done_count = ack_job(redis_conn, uid, job_id)
            logger.info(f"[{done_count}/{total_count}] processing...")
            continue

        if len(images) == 0:
            logger.warning(f"No images found for {object_id}")
            done_count = ack_job(redis_conn, uid, job_id)
            logger.info(f"[{done_count}/{total_count}] processing...")
            continue

        if not os.path.exists(queries_path):
            logger.warning(f"No queries found for {object_id}")
            continue

        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir, exist_ok=True)

        if isinstance(images[0], str):
            images = [
                Image.open(image_path).convert("RGB") for image_path in images
            ]

        with open(queries_path, "r") as f:
            data = json.load(f)[0]
        class_name, queries = data["class_name"], data["queries"]

        with timer:
            points_all = molmo.process(images, queries, image_wh)

        # save points
        np.save(points_path, points_all)
        done_count = ack_job(redis_conn, uid, job_id)
        logger.info(
            f"[{done_count}/{total_count}] processing... "
            f"last iteration takes {timer.single_time:.3f}s, "
            f"on average {timer.fps:.3f} it/s, "
            f"ETA: {(total_count - done_count) / timer.fps:.3f}s"
        )


if __name__ == "__main__":
    import fire

    fire.Fire(run_stage2_molmo)
