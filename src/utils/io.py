import json
import os
import re
import tarfile
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_mesh_kaolin(mesh_path):
    import kaolin

    """
    Load a 3D mesh using Kaolin.

    Args:
        mesh_path (str): Path to the mesh file (supports various formats)

    Returns:
        tuple: Vertices and faces of the loaded mesh
    """
    try:
        # Load the mesh using Kaolin
        mesh = kaolin.io.gltf.import_mesh(mesh_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh with Kaolin: {e}")
        return None


def read_mesh_trimesh(mesh_path):
    """
    Load a 3D mesh from a GLB file using trimesh.

    Args:
        glb_path (str): Path to the GLB file

    Returns:
        trimesh.Trimesh or trimesh.Scene: The loaded mesh or scene
    """
    try:
        # Load the GLB file
        mesh = trimesh.load_mesh(mesh_path, file_type="glb")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        return mesh
    except Exception as e:
        print(f"Error loading mesh with trimesh: {e}")
        return None


def read_image(image_path_or_bytes):
    if isinstance(image_path_or_bytes, str):
        image = cv2.imread(image_path_or_bytes, cv2.IMREAD_COLOR_RGB)
    elif hasattr(image_path_or_bytes, "read"):
        image_bytes = image_path_or_bytes.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR_RGB)
    else:
        raise ValueError("Invalid input type for load_image")

    return image


def read_image_pil(image_path_or_bytes):
    if isinstance(image_path_or_bytes, str):
        image = Image.open(image_path_or_bytes).convert("RGB")
    elif hasattr(image_path_or_bytes, "read"):
        image_bytes = BytesIO(image_path_or_bytes.read())
        image = Image.open(image_bytes).convert("RGB")
    return image


def read_image_from_tarfile(
    tar_path: str,
    num_images: Optional[int] = None,
    image_skip: Optional[int] = None,
    load_depth: bool = False,
    load_cam_params: bool = False,
):
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

        if num_images is not None and num_images > 0:
            samples = samples[:num_images]
        if image_skip is not None and image_skip > 0:
            samples = samples[::image_skip]

        # load images
        images = []
        image_ids = []
        for sample in samples:
            with tar.extractfile(sample["image"]) as f:
                image = read_image_pil(f)
                images.append(image)
                image_ids.append(sample["image"].name)
        return_dict = dict(images=images, image_ids=image_ids)

        if load_depth:
            return_dict["depths"] = []
            for sample in samples:
                with tar.extractfile(sample["depth"]) as f:
                    depth = read_depth(f)
                    return_dict["depths"].append(depth)

        if load_cam_params:
            return_dict["c2ws"] = []
            return_dict["Ks"] = []
            for sample in samples:
                with tar.extractfile(sample["json"]) as f:
                    c2w, K = read_cam_params(f, images[0].size)
                    return_dict["c2ws"].append(c2w)
                    return_dict["Ks"].append(K)

    return return_dict


def read_exr(exr_path_or_bytes):
    """
    Load a normal and depth map from an EXR file.

    Args:
        exr_path (str): Path to the EXR file

    Returns:
        tuple: Normal and depth map
    """
    if isinstance(exr_path_or_bytes, str):
        normal_d = cv2.imread(exr_path_or_bytes, cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )
    elif hasattr(exr_path_or_bytes, "read"):
        image_bytes = exr_path_or_bytes.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        normal_d = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )
    else:
        raise ValueError("Invalid input type for load_exr")

    normal, depth = normal_d[..., :3], normal_d[..., 3]
    return normal, depth


def read_depth(exr_path_or_bytes):
    normal, depth = read_exr(exr_path_or_bytes)
    return depth


def read_cam_params(json_file_or_bytes, image_wh: Tuple[int, int]):
    """
    Read camera parameters from a JSON file.

    Args:
        json_file (str): Path to the JSON file
        image_wh (tuple): Shape of the image (height, width)

    Returns:
        tuple: Camera pose (c2w) and intrinsic matrix (K)
    """
    if isinstance(json_file_or_bytes, str):
        with open(json_file_or_bytes, "r", encoding="utf8") as reader:
            json_content = json.load(reader)
    else:
        json_content = json.load(json_file_or_bytes)

    c2w = np.eye(4)
    c2w[:3, 0] = np.array(json_content["x"])
    c2w[:3, 1] = np.array(json_content["y"])
    c2w[:3, 2] = np.array(json_content["z"])
    c2w[:3, 3] = np.array(json_content["origin"])
    swap_flip = np.array(
        [
            [1, 0, 0, 0],  # x stays x
            [0, 0, 1, 0],  # y becomes z
            [0, -1, 0, 0],  # z becomes -y
            [0, 0, 0, 1],  # homogeneous coordinate
        ],
        dtype=c2w.dtype,
    )
    c2w_transformed = swap_flip @ c2w

    fov = json_content["x_fov"]
    fx = image_wh[0] / 2 / np.tan(fov / 2)
    fy = image_wh[1] / 2 / np.tan(fov / 2)
    K = np.array(
        [
            [fx, 0, (image_wh[0]) / 2],
            [0, fy, (image_wh[1]) / 2],
            [0, 0, 1],
        ]
    )

    return c2w_transformed, K
