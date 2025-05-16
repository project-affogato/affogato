import json
import os
import time
from typing import Optional

import redis

from src.dataset.objaverse.dataset import ObjaverseDatasetTar
from src.job_manager import ack_job, fetch_next_job, generate_unique_id
from src.model.vllm.gemma3 import Gemma3
from src.utils.logging import get_logger
from src.utils.timer import Timer

logger = get_logger(__name__)


def run_stage1_gemma3(
    uid: str,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    overwrite: bool = False,
    redis_host: str = "141.223.85.85",
    redis_port: int = 6379,
):
    logger.info("==> Running stage 1: gemma3")

    redis_conn = redis.Redis(host=redis_host, port=redis_port)
    logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

    # Fetch hyperparameters from Redis
    config_json = redis_conn.hget(f"experiment:{uid}", "config")
    if not config_json:
        raise ValueError(f"No config found for {uid}")

    hparams = json.loads(config_json)
    logger.info(f"Fetched hyperparameters: {hparams}")

    required_keys = ["subset", "filename", "prompt_file"]
    assert all(key in hparams for key in required_keys), (
        f"Missing required keys: {required_keys}"
    )
    total_count = int(redis_conn.hget(f"experiment:{uid}", "total_count"))

    dataset = ObjaverseDatasetTar(
        data_dir=data_dir or hparams["data_dir"],
        subset=hparams["subset"],
        render_dir="gobjaverse_reduced_images",
        load_image_only=True,
        num_images=25,
        image_skip=5,
    )
    gemma3 = Gemma3(use_tqdm=False, prompt_file=hparams["prompt_file"])

    consumer_id = f"worker-{generate_unique_id()}"
    logger.info(f"Worker ID: {consumer_id}, Job ID: {uid}")

    save_dir = save_dir or hparams["save_dir"]
    filename = hparams["filename"]

    timer = Timer()
    while True:
        # for i in tqdm(range(len(dataset))):
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
        object_id = sample["object_id"]
        instance_dir = os.path.join(save_dir, object_id)
        queries_path = os.path.join(instance_dir, filename)

        if os.path.exists(queries_path) and not overwrite:
            logger.warning(f"Skipping {object_id} because it already exists")
            done_count = ack_job(redis_conn, uid, job_id)
            logger.info(f"[{done_count}/{total_count}] processing...")
            continue

        if len(images) == 0:
            logger.warning(f"No images found for {object_id}")
            done_count = ack_job(redis_conn, uid, job_id)
            logger.info(f"[{done_count}/{total_count}] processing...")
            continue

        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir, exist_ok=True)

        with timer:
            queries = gemma3.process(images)

        # save queries
        try:
            with open(queries_path, "w") as f:
                json.dump(queries, f)
        except Exception as e:
            logger.warning(f"Error saving queries for {object_id}: {e}")
            continue

        done_count = ack_job(redis_conn, uid, job_id)
        logger.info(
            f"[{done_count}/{total_count}] processing... "
            f"last iteration takes {timer.single_time:.3f}s, "
            f"on average {timer.fps:.3f} it/s, "
            f"ETA: {(total_count - done_count) / timer.fps:.3f}s"
        )


if __name__ == "__main__":
    import fire

    fire.Fire(run_stage1_gemma3)
