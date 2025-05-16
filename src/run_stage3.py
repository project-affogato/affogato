import json
import os
import time
from typing import Literal, Optional

import numpy as np
import redis

from src.dataset.objaverse.dataset import ObjaverseDatasetTar
from src.job_manager import ack_job, fetch_next_job, generate_unique_id
from src.utils.geometry import PointCloudToImageMapper
from src.utils.logging import get_logger
from src.utils.timer import Timer

logger = get_logger(__name__)


def instantiate_sam(
    backend: Literal["sam2", "mobilesam"] = "sam2",
    mask_selection_mode: Literal[
        "smallest_mask", "highest_score", "random"
    ] = "smallest_mask",
):
    if backend == "sam2":
        from src.model.sam2 import SAM2

        return SAM2(mask_selection_mode=mask_selection_mode)
    elif backend == "mobilesam":
        from src.model.mobilesam import MobileSAM

        return MobileSAM(mask_selection_mode=mask_selection_mode)
    else:
        raise ValueError(f"Invalid backend: {backend}")


def run_stage3_sam(
    uid: str,
    data_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    backend: Literal["sam2", "mobilesam"] = "mobilesam",
    overwrite: bool = False,
    redis_host: str = "141.223.85.85",
    redis_port: int = 6379,
):
    logger.info("==> Running stage 3: sam2")

    redis_conn = redis.Redis(host=redis_host, port=redis_port)
    logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

    config_json = redis_conn.hget(f"experiment:{uid}", "config")
    if not config_json:
        raise ValueError(f"No config found for {uid}")

    hparams = json.loads(config_json)
    logger.info(f"Fetched hyperparameters: {hparams}")

    required_keys = [
        "subset",
        "filename",
        "point_filename",
        "num_images",
        "image_skip",
        "mask_selection_mode",
    ]
    assert all(key in hparams for key in required_keys), (
        f"Missing required keys: {required_keys}"
    )
    total_count = int(redis_conn.hget(f"experiment:{uid}", "total_count"))

    dataset = ObjaverseDatasetTar(
        data_dir=data_dir or hparams["data_dir"],
        subset=hparams["subset"],
        render_dir="gobjaverse_reduced",
        load_image_only=False,
        num_images=int(hparams["num_images"]),
        image_skip=int(hparams["image_skip"]),
    )
    sam = instantiate_sam(
        backend=backend,
        mask_selection_mode=hparams["mask_selection_mode"],
    )

    consumer_id = f"worker-{generate_unique_id()}"
    logger.info(f"Worker ID: {consumer_id}, Job ID: {uid}")

    save_dir = save_dir or hparams["save_dir"]
    point_filename = hparams["point_filename"]
    filename = hparams["filename"]

    timer = Timer()
    while True:
        job_id, job = fetch_next_job(redis_conn, uid, consumer_id)
        if job is None:
            logger.info("No job found, sleeping for 1 second")
            time.sleep(1)
            continue

        idx = int(job["idx"])
        object_id = dataset.object_ids[idx]
        instance_dir = os.path.join(save_dir, object_id)
        points_path = os.path.join(instance_dir, point_filename)
        mask_path = os.path.join(instance_dir, filename)

        # Skip if file already exists
        if os.path.exists(mask_path) and not overwrite:
            logger.debug(f"Skipping {object_id} because mask already exists")
            done_count = ack_job(redis_conn, uid, job_id)
            logger.info(f"[{done_count}/{total_count}] processing...")
            continue

        try:
            sample = dataset[idx]
        except Exception as e:
            logger.warning(f"Error processing {dataset.object_ids[idx]}: {e}")
            continue
        images = sample["images"]
        depths = sample["depths"]
        c2ws = sample["c2ws"]
        Ks = sample["Ks"]
        xyz = sample["points"]

        if len(images) == 0:
            logger.warning(f"No images found for {object_id}")
            done_count = ack_job(redis_conn, uid, job_id)
            logger.info(f"[{done_count}/{total_count}] processing...")
            continue

        if not os.path.exists(points_path):
            logger.warning(f"No points found for {object_id}")
            continue

        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir, exist_ok=True)

        try:
            points_all = np.load(points_path)
        except Exception as e:
            logger.warning(f"Error loading points for {object_id}: {e}")
            continue

        num_images = points_all.shape[1]
        mapper = PointCloudToImageMapper(
            depths[0].shape[:2], intrinsics=Ks[0], visibility_threshold=0.01
        )

        num_querise = points_all.shape[0]
        heatmap_3d = np.zeros((num_querise, len(xyz)))
        counts = np.zeros((num_querise, len(xyz)))
        heatmap_2d_all = []
        with timer:
            for i in range(num_images):
                image = images[i]
                masks, scores, logits = sam.process(image, points_all[:, i])

                heatmap_2d = 1 / (1 + np.exp(-logits))
                mapping = mapper.compute_mapping(c2ws[i], xyz, depth=depths[i])
                heatmap_2d_3d = heatmap_2d[:, mapping[:, 0], mapping[:, 1]]
                valid_mapping_mask = mapping[:, 2] != 0
                counts[:, valid_mapping_mask] += 1
                heatmap_3d[:, valid_mapping_mask] += heatmap_2d_3d[
                    :, valid_mapping_mask
                ]
                heatmap_2d_all.append(heatmap_2d)

        counts[counts == 0] = 1e-5
        heatmap_3d = heatmap_3d / counts

        # save masks
        xyzc = np.concatenate(
            [xyz, np.transpose(heatmap_3d, (1, 0))], axis=1
        ).astype(np.float16)
        np.save(mask_path, xyzc)
        np.savez_compressed(
            os.path.join(instance_dir, "mask_2d.npz"),
            mask_2d=np.stack(heatmap_2d_all).astype(bool),
        )

        done_count = ack_job(redis_conn, uid, job_id)
        logger.info(
            f"[{done_count}/{total_count}] processing... "
            f"last iteration takes {timer.single_time:.3f}s, "
            f"on average {timer.fps:.3f} it/s, "
            f"ETA: {(total_count - done_count) / timer.fps:.3f}s"
        )


if __name__ == "__main__":
    import fire

    fire.Fire(run_stage3_sam)
