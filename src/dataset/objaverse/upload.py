from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()

data_dir = Path("/datasets/objaverse/packaged_dataset")
tarfiles = sorted(data_dir.glob("*.tar.gz"))

for tarfile in tarfiles:
    api.upload_file(
        repo_id="project-affogato/affogato",
        repo_type="dataset",
        path_or_fileobj=tarfile,
        path_in_repo=tarfile.name,
    )
