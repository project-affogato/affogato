import json
import os
import re
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Optional

from natsort import natsorted
from torch.utils.data import Dataset

from src.utils.geometry import normalize_mesh, sample_points_on_mesh
from src.utils.io import (
    read_cam_params,
    read_depth,
    read_image,
    read_image_from_tarfile,
    read_mesh_trimesh,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ObjaverseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        subset: str = "Daily-Used",
        render_dir: str = "gobjaverse",
        load_image_only: bool = False,
        num_points: Optional[int] = 1024 * 16,
        num_images: Optional[int] = None,
        image_skip: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.render_dir = render_dir
        self.num_points = num_points
        self.load_image_only = load_image_only
        self.num_images = num_images
        self.image_skip = image_skip
        self.rank = rank
        self.world_size = world_size

        # load object_ids
        if subset.lower() == "all":
            subset_files = [
                self.data_dir / render_dir / f"gobjaverse_280k_{subset}.json"
                for subset in ["Furnitures", "Daily-Used", "Transportations", "Electronics"]
            ]
            object_ids = []
            for subset_file in subset_files:
                with open(subset_file, "r") as f:
                    object_ids.extend(json.load(f))
        else:
            with open(
                self.data_dir / render_dir / f"gobjaverse_280k_{subset}.json",
                "r",
            ) as f:
                object_ids = natsorted(json.load(f))

        # load mapping
        mapping_file = (
            self.data_dir
            / render_dir
            / "gobjaverse_280k_index_to_objaverse.json"
        )
        with open(mapping_file, "r") as f:
            self.mapping = json.load(f)

        self.object_ids = [
            self.mapping[uid][: -len(".glb")] for uid in object_ids
        ]
        if self.rank is not None and self.world_size is not None:
            self.object_ids = self.object_ids[self.rank :: self.world_size]

        logger.info(
            f"{self.__class__.__name__}: {len(self.object_ids)} objects from {subset}. "
        )

    def __len__(self):
        return len(self.object_ids)

    def filter_existing(self, save_dir: str, filename: str):
        len_orig = len(self)
        existing_ids = []
        for uid in self.object_ids:
            points_path = os.path.join(save_dir, uid, filename)
            if os.path.exists(points_path):
                existing_ids.append(uid)
        logger.warning(f"Found {len(existing_ids)} existing objects.")

        new_ids = sorted(list(set(self.object_ids) - set(existing_ids)))
        self.object_ids = new_ids
        logger.warning(
            f"Reduced dataset from {len_orig} to {len(self)} objects."
        )

    def load_images_from_tarfile(self, tar_path: str):
        with tarfile.open(tar_path, "r:*") as tar:
            members = tar.getmembers()
            # Group files by sample ID and ensure all required files exist
            samples = {}
            for m in members:
                fname = os.path.basename(m.name)
                if fname.endswith(".json"):
                    sid = fname[:-5]
                    samples.setdefault(sid, {})["json"] = m
                elif fname.endswith("_nd.exr"):
                    sid = fname[: -len("_nd.exr")]
                    samples.setdefault(sid, {})["depth"] = m
                elif re.match(r"\d{5}\.png$", fname):
                    sid = fname[:-4]
                    samples.setdefault(sid, {})["image"] = m

            # Sort by sample ID and convert to array to ensure consistent ordering
            samples = [files for _, files in sorted(samples.items())]

            # load images
            images = []
            for sample in samples:
                with tar.extractfile(sample["image"]) as f:
                    image = read_image(f)
                    images.append(image)
        return images

    def get_by_object_id(self, object_id: str):
        index = self.object_ids.index(object_id)
        return self[index]

    def __getitem__(self, index: int):
        object_id = self.object_ids[index]

        object_dir = self.data_dir / self.render_dir / object_id
        if (
            not object_dir.is_dir()
            and object_dir.with_suffix(".tar").is_file()
        ):
            images = self.load_images_from_tarfile(
                object_dir.with_suffix(".tar")
            )
            image_hw = images[0].shape[:2]
        else:
            image_paths = sorted(
                (self.data_dir / self.render_dir / object_id).glob("*.png")
            )
            images = [str(image_path) for image_path in image_paths]
            if len(images) == 0:
                image_hw = (512, 512)
            else:
                image_hw = read_image(images[0]).shape[:2]

        if self.num_images is not None and self.num_images > 0:
            image_paths = image_paths[: self.num_images]
            images = images[: self.num_images]

        data_dict = dict(
            images=images,
            image_wh=image_hw[::-1],
            object_id=object_id,
        )
        if self.load_image_only:
            return data_dict

        # load mesh
        mesh_path = (
            self.data_dir / "hf-objaverse-v1" / "glbs" / f"{object_id}.glb"
        )
        cam_paths = [
            str(image_path).replace(".png", ".json")
            for image_path in image_paths
        ]
        depth_paths = [
            str(image_path).replace(".png", "_nd.exr")
            for image_path in image_paths
        ]
        mesh = read_mesh_trimesh(mesh_path)
        scale, offset = normalize_mesh(mesh)
        new_mesh = deepcopy(mesh)
        new_mesh.vertices = scale * (new_mesh.vertices + offset)
        new_points, _ = sample_points_on_mesh(new_mesh, self.num_points)

        depths = [read_depth(depth_path) for depth_path in depth_paths]
        # Use a single iteration over cam_paths and unzip the results
        c2ws, Ks = zip(
            *[read_cam_params(cam_path, image_hw) for cam_path in cam_paths]
        )
        c2ws, Ks = list(c2ws), list(Ks)

        data_dict.update(
            dict(
                mesh=new_mesh,
                points=new_points,
                depths=depths,
                c2ws=c2ws,
                Ks=Ks,
            )
        )
        return data_dict


class ObjaverseDatasetTar(ObjaverseDataset):
    def __getitem__(self, index: int):
        object_id = self.object_ids[index]

        object_dir = self.data_dir / self.render_dir / object_id
        tar_path = object_dir.with_suffix(".tar.gz")
        assert tar_path.is_file(), f"Tar file not found for {object_id}"

        data_dict = read_image_from_tarfile(
            tar_path,
            load_depth=not self.load_image_only,
            load_cam_params=not self.load_image_only,
            num_images=self.num_images,
            image_skip=self.image_skip,
        )
        image_wh = data_dict["images"][0].size

        data_dict["image_wh"] = image_wh
        data_dict["object_id"] = object_id

        # load mesh
        if not self.load_image_only:
            mesh_path = (
                self.data_dir / "hf-objaverse-v1" / "glbs" / f"{object_id}.glb"
            )
            mesh = read_mesh_trimesh(mesh_path)
            scale, offset = normalize_mesh(mesh)
            new_mesh = deepcopy(mesh)
            new_mesh.vertices = scale * (new_mesh.vertices + offset)
            new_points, _ = sample_points_on_mesh(new_mesh, self.num_points)
            data_dict["mesh"] = new_mesh
            data_dict["points"] = new_points

        return data_dict


if __name__ == "__main__":
    import numpy as np

    dataset = ObjaverseDatasetTar(
        data_dir="/datasets/objaverse",
        subset="Furnitures",
        render_dir="gobjaverse_reduced",
        load_image_only=False,
        num_points=1024 * 16,
        num_images=25,
        image_skip=2,
    )
    sample0 = dataset[0]
    sample1 = dataset[1]

    c2w0 = np.stack(sample0["c2ws"], axis=0)
    c2w1 = np.stack(sample1["c2ws"], axis=0)

    diff = c2w0 @ np.linalg.inv(c2w1)
    import pdb

    pdb.set_trace()
