import http.client
import os
import random
import subprocess
import tempfile
import time
import urllib.request
from multiprocessing import Pool
from typing import Callable, Literal, Optional, Tuple

import fsspec
import objaverse.xl as oxl
from objaverse.utils import get_file_hash
from tqdm import tqdm

from src.dataset.objaverse.utils import get_dataset_info
from src.utils.logging import get_logger

logger = get_logger(__name__)

GOBJAVERSE_REMOTE_URL = (
    "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d"
)


class CustomObjaverseDownloader(oxl.SketchfabDownloader):
    @classmethod
    def _download_object(
        cls,
        file_identifier: str,
        hf_object_path: str,
        download_dir: Optional[str],
        expected_sha256: str,
        handle_found_object: Optional[Callable] = None,
        handle_modified_object: Optional[Callable] = None,
    ) -> Tuple[str, Optional[str]]:
        """Download the object for the given uid.

        Args:
            file_identifier: The file identifier of the object.
            hf_object_path: The path to the object in the Hugging Face repo. Here,
                hf_object_path is the part that comes after "main" in the Hugging Face
                repo url:
                https://huggingface.co/datasets/allenai/objaverse/resolve/main/{hf_object_path}
            download_dir: The base directory to download the object to. Supports all
                file systems supported by fsspec. Defaults to "~/.objaverse".
            expected_sha256 (str): The expected SHA256 of the contents of the downloade
                object.
            handle_found_object (Optional[Callable]): Called when an object is
                successfully found and downloaded. Here, the object has the same sha256
                as the one that was downloaded with Objaverse-XL. If None, the object
                will be downloaded, but nothing will be done with it. Args for the
                function include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): GitHub URL of the 3D object.
                - sha256 (str): SHA256 of the contents of the 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used.
            handle_modified_object (Optional[Callable]): Called when a modified object
                is found and downloaded. Here, the object is successfully downloaded,
                but it has a different sha256 than the one that was downloaded with
                Objaverse-XL. This is not expected to happen very often, because the
                same commit hash is used for each repo. If None, the object will be
                downloaded, but nothing will be done with it. Args for the function
                include:
                - local_path (str): Local path to the downloaded 3D object.
                - file_identifier (str): GitHub URL of the 3D object.
                - new_sha256 (str): SHA256 of the contents of the newly downloaded 3D
                    object.
                - old_sha256 (str): Expected SHA256 of the contents of the 3D object as
                    it was when it was downloaded with Objaverse-XL.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used.
            handle_missing_object (Optional[Callable]): Called when an object that is in
                Objaverse-XL is not found. Here, it is likely that the repository was
                deleted or renamed. If None, nothing will be done with the missing
                object. Args for the function include:
                - file_identifier (str): GitHub URL of the 3D object.
                - sha256 (str): SHA256 of the contents of the original 3D object.
                - metadata (Dict[str, Any]): Metadata about the 3D object, including the
                    GitHub organization and repo names.
                Return is not used.


        Returns:
            A tuple of the uid and the path to where the downloaded object. If
            download_dir is None, the path will be None.
        """
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{hf_object_path}"

        with tempfile.TemporaryDirectory() as temp_dir:
            # download the file locally
            temp_path = os.path.join(temp_dir, hf_object_path)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            temp_path_tmp = f"{temp_path}.tmp"

            # Add retry logic for downloading
            max_retries = 8  # Increased max retries
            base_retry_delay = 5  # seconds

            for attempt in range(max_retries):
                try:
                    # Add a small random delay before each request to avoid synchronized requests
                    if attempt > 0:
                        jitter = random.uniform(0.1, 0.5)
                        time.sleep(jitter)

                    with open(temp_path_tmp, "wb") as file:
                        with urllib.request.urlopen(hf_url) as response:
                            # Read in chunks to handle large files better
                            chunk_size = 8192
                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break
                                file.write(chunk)

                    # If we get here, download was successful
                    os.rename(temp_path_tmp, temp_path)
                    break

                except urllib.error.HTTPError as e:
                    if e.code == 429:  # Too Many Requests
                        # Exponential backoff with jitter for rate limiting
                        retry_delay = base_retry_delay * (
                            2**attempt
                        ) + random.uniform(1, 5)
                        logger.warning(
                            f"Rate limit hit (429) for {file_identifier}. Waiting {retry_delay:.1f}s before retry {attempt + 1}/{max_retries}"
                        )
                        time.sleep(retry_delay)
                        continue
                    elif attempt < max_retries - 1:
                        logger.warning(
                            f"HTTP error {e.code} on attempt {attempt + 1} for {file_identifier}: {e}"
                        )
                        time.sleep(base_retry_delay * (attempt + 1))
                    else:
                        logger.error(
                            f"Failed to download {file_identifier} after {max_retries} attempts: {e}"
                        )
                        raise
                except (
                    urllib.error.URLError,
                    ConnectionResetError,
                    http.client.IncompleteRead,
                ) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Download attempt {attempt + 1} failed for {file_identifier}: {e}"
                        )
                        time.sleep(
                            base_retry_delay * (attempt + 1)
                        )  # Exponential backoff
                    else:
                        logger.error(
                            f"Failed to download {file_identifier} after {max_retries} attempts: {e}"
                        )
                        raise

            # get the sha256 of the downloaded file
            sha256 = get_file_hash(temp_path)

            if sha256 == expected_sha256:
                if handle_found_object is not None:
                    handle_found_object(
                        local_path=temp_path,
                        file_identifier=file_identifier,
                        sha256=sha256,
                        metadata={},
                    )
            else:
                if handle_modified_object is not None:
                    handle_modified_object(
                        local_path=temp_path,
                        file_identifier=file_identifier,
                        new_sha256=sha256,
                        old_sha256=expected_sha256,
                        metadata={},
                    )

            if download_dir is not None:
                filename = os.path.join(
                    download_dir, "hf-objaverse-v1", hf_object_path
                )
                fs, path = fsspec.core.url_to_fs(filename)
                fs.makedirs(os.path.dirname(path), exist_ok=True)
                fs.put(temp_path, path)
            else:
                path = None

        return file_identifier, path


def download_gobjaverse_metadata(download_dir):
    download_files = [
        "gobjaverse_280k_index_to_objaverse.json",
        "gobjaverse_280k_split/gobjaverse_280k_Daily-Used.json",
        "gobjaverse_280k_split/gobjaverse_280k_Furnitures.json",
        "gobjaverse_280k_split/gobjaverse_280k_Transportations.json",
        "gobjaverse_280k_split/gobjaverse_280k_Electronics.json",
    ]
    for file in download_files:
        local_file_path = os.path.join(download_dir, os.path.basename(file))
        if not os.path.exists(local_file_path):
            logger.info(
                "Downloading {} from {}".format(file, GOBJAVERSE_REMOTE_URL)
            )
            os.system(
                "wget -P {} {}".format(
                    download_dir, os.path.join(GOBJAVERSE_REMOTE_URL, file)
                )
            )
        else:
            logger.info(
                "{} already exists at {}".format(file, local_file_path)
            )


def download_gobjaverse_rendering(download_dir, gobjaverse_id, object_id):
    remote_url = os.path.join(
        GOBJAVERSE_REMOTE_URL,
        "objaverse_tar",
        gobjaverse_id + ".tar",
    )
    local_path = os.path.join(download_dir, object_id + ".tar")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Skip if already downloaded
    if os.path.exists(local_path) or os.path.exists(local_path + ".gz"):
        logger.info(f"{local_path} already exists; skipping download")
        return
    if os.path.exists(local_path[: -len(".tar")]):
        logger.info(
            f"{local_path[: -len('.tar')]} already exists; skipping download"
        )
        return

    # Use subprocess with quiet mode to suppress wget output
    subprocess.run(
        ["wget", "-q", remote_url, "-O", local_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def download_gobjaverse_rendering_wrapper(args):
    return download_gobjaverse_rendering(*args)


def download(
    download_dir: str,
    subset: Literal[
        "Daily-Used", "Furnitures", "Transportations", "Electronics"
    ],
    limit: Optional[int] = None,
):
    """Download 3D objects from Objaverse and GObjaverse.

    Args:
        download_dir: Directory to download files to
        subset: Category of objects to download
        limit: Maximum number of objects to download
    """
    # Download metadata
    gobjaverse_dir = os.path.join(download_dir, "gobjaverse")
    os.makedirs(gobjaverse_dir, exist_ok=True)
    download_gobjaverse_metadata(gobjaverse_dir)

    # Load metadata
    dataset_info = get_dataset_info(download_dir, subset)
    object_ids = dataset_info["object_id"].tolist()

    def download_with_retry():
        try:
            CustomObjaverseDownloader.download_objects(
                dataset_info,
                download_dir=download_dir,
                processes=min(16, os.cpu_count()),
            )
            return  # Success, exit the function
        except Exception as e:
            logger.warning(f"Download failed with error: {e}")
            logger.info("Retrying download in 1 seconds...")
            time.sleep(1)

    download_with_retry()
    # Download renderings in parallel
    logger.info(f"Downloading {len(object_ids)} GObjaverse renderings")

    tasks = [
        (
            os.path.join(download_dir, "gobjaverse"),
            series.gobjaverse_id,
            series.object_id,
        )
        for _, series in dataset_info.iterrows()
    ]

    with Pool(min(16, os.cpu_count())) as p:
        with tqdm(
            total=len(tasks), desc="Downloading renderings", unit="file"
        ) as pbar:
            for _ in p.imap_unordered(
                download_gobjaverse_rendering_wrapper, tasks
            ):
                pbar.update()


if __name__ == "__main__":
    import fire

    fire.Fire(download)
