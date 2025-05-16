import json
from pathlib import Path

import pandas as pd
from objaverse.xl.sketchfab import SketchfabDownloader

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_dataset_info(data_dir: str, subset: str):
    data_dir = Path(data_dir)

    # get objaverse annotations
    objaverse_df = SketchfabDownloader.get_annotations(download_dir=data_dir)
    objaverse_df["uid"] = objaverse_df["fileIdentifier"].str.extract(
        r"/3d-models/([^-/]+)"
    )

    # get g-objaverse annotations
    metadata_dir = data_dir / "gobjaverse"
    if not metadata_dir.exists():
        metadata_dir = data_dir / "gobjaverse_reduced"
    if not metadata_dir.exists():
        metadata_dir = data_dir / "gobjaverse_reduced_images"
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Metadata directory {metadata_dir} does not exist"
        )

    # get gobjaverse annotations
    with open(metadata_dir / "gobjaverse_280k_index_to_objaverse.json") as f:
        id_to_objaverse = json.load(f)

    gobjaverse_df = pd.DataFrame(
        list(id_to_objaverse.items()),
        columns=["gobjaverse_id", "glb_path"],
    )
    gobjaverse_df["uid"] = gobjaverse_df["glb_path"].str.extract(
        r"/([a-f0-9]+)\.glb"
    )
    # merged
    merged_df = pd.merge(objaverse_df, gobjaverse_df, on="uid", how="inner")

    # load subset ids
    with open(metadata_dir / f"gobjaverse_280k_{subset}.json") as f:
        gobjaverse_ids = json.load(f)

    matched_df = merged_df[
        merged_df["gobjaverse_id"].isin(gobjaverse_ids)
    ].copy()
    matched_df.loc[:, "object_id"] = matched_df["glb_path"].str.replace(
        ".glb", ""
    )
    logger.info(f"Found {len(matched_df)} matches")
    return matched_df
