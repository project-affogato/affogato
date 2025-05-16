import json
import random
import string

import redis

from src.dataset.objaverse.dataset import ObjaverseDataset
from src.utils.logging import get_logger

CONSUMER_GROUP = "mygroup"

logger = get_logger(__name__)


def generate_unique_id(length=16):
    """Generate a unique ID of specified length."""
    characters = string.ascii_letters + string.digits  # Pool of characters
    unique_id = "".join(random.choices(characters, k=length))
    return unique_id


def create_job(
    job_name: str,
    host: str = "141.223.85.85",
    data_dir: str = "/datasets/objaverse",
    subset: str = "Furnitures",
    save_dir: str = "/datasets/objaverse/annotations",
    num_images: int = 25,
    image_skip: int = 1,
    **hyperparams,
):
    logger.info(f"Creating job for {job_name} with {subset} subset")
    redis_conn = redis.Redis(host=host, port=6379, db=0)
    logger.info(f"Connected to Redis at {host}:6379")

    uid = generate_unique_id()

    dataset = ObjaverseDataset(
        data_dir,
        subset,
        render_dir="gobjaverse_reduced",
        load_image_only=True,
    )
    total_count = len(dataset)
    logger.info(f"Total count: {total_count}")

    # Initialize job queue
    job_queue = f"queue:{uid}"
    logger.info(f"Job queue: {job_queue}")
    jobs = [{"idx": idx} for idx in range(len(dataset))]

    BATCH_SIZE = 1000
    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i : i + BATCH_SIZE]
        pipe = redis_conn.pipeline()
        for job in batch:
            pipe.xadd(job_queue, job)
        pipe.execute()

    # Save hyperparameters in Redis
    hyperparams_key = f"experiment:{uid}"
    logger.info(f"Saving hyperparameters to {hyperparams_key}")
    hyperparams.update(
        {
            "job_name": job_name,
            "data_dir": data_dir,
            "subset": subset,
            "save_dir": save_dir,
            "num_images": num_images,
            "image_skip": image_skip,
        }
    )

    # Store hyperparameters as hash
    redis_conn.hset(
        hyperparams_key,
        mapping={
            "config": json.dumps(hyperparams),
            "total_count": total_count,
            "done_count": 0,
        },
    )

    # Set up consumer group
    redis_conn.xgroup_create(job_queue, CONSUMER_GROUP, id="0")

    return uid


def fetch_next_job(
    redis_conn,
    uid: str,
    consumer_id: str,
    block: int = 1000,
    idle_ms: int = 60000,
):
    """
    Fetch the next available job from the stream.

    Attempts to get a new message first, then tries to claim an idle pending job.

    Args:
        uid: Experiment unique ID
        consumer_id: ID of the consumer requesting the job
        block: How long to block waiting for new messages (ms)
        idle_ms: How long a job must be idle before it can be claimed (ms)

    Returns:
        Tuple of (message_id, fields) if job found, None otherwise
    """
    stream_id = f"queue:{uid}"

    # Try getting a new message first
    xread_output = redis_conn.xreadgroup(
        groupname=CONSUMER_GROUP,
        consumername=consumer_id,
        streams={stream_id: ">"},
        count=1,
        block=block,
    )

    if xread_output:
        _, messages = xread_output[0]
        message_id, fields = messages[0]
        fields = {
            k.decode("utf-8"): v.decode("utf-8") for k, v in fields.items()
        }
        return message_id, fields

    # Try claiming an idle message
    try:
        res = redis_conn.xautoclaim(
            name=stream_id,
            groupname=CONSUMER_GROUP,
            consumername=consumer_id,
            min_idle_time=idle_ms,
            start_id="0-0",
            count=1,
        )
    except redis.exceptions.ResponseError:
        return None, None

    # Check if we got any results
    if not res or not res[1]:
        return None, None

    # Extract job details
    job_id, fields = res[1][0]
    # Decode byte keys and values to strings
    fields = {k.decode("utf-8"): v.decode("utf-8") for k, v in fields.items()}
    return job_id, fields


def ack_job(redis_conn, uid: str, job_id: str):
    stream_id = f"queue:{uid}"
    xack_output = redis_conn.xack(stream_id, CONSUMER_GROUP, job_id)
    redis_conn.hincrby(f"experiment:{uid}", "done_count", 1)
    logger.debug(f"Acknowledged job {job_id}: {xack_output}")
    done_count = get_done_count(redis_conn, uid)
    return done_count


def get_done_count(redis_conn, uid: str) -> int:
    """
    Get the current count of completed jobs for a given experiment.

    Args:
        redis_conn: Redis connection object
        uid: Unique identifier for the experiment

    Returns:
        int: Number of completed jobs
    """
    done_count = redis_conn.hget(f"experiment:{uid}", "done_count")
    if done_count is None:
        return 0
    return int(done_count)


if __name__ == "__main__":
    import fire

    fire.Fire(create_job)
