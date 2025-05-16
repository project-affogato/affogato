from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.dataset.objaverse.dataset import ObjaverseDatasetTar
from src.utils.logging import get_logger

logger = get_logger(__name__)


def render_heatmap_batch_torch(
    points: torch.Tensor,  # (N,3)
    heatmap: torch.Tensor,  # (N,C)
    c2w: torch.Tensor,  # (B,4,4)
    K: torch.Tensor,  # (B,3,3)
    depth_img: torch.Tensor,  # (B,H,W)
    image_shape: tuple[int, int],
    sigma: float = 1.0,
    depth_tol: float = 0.01,
    radius: int = 1,
) -> torch.Tensor:
    """
    points    : (N,3) world coords
    heatmap   : (N,C) per-point colors
    c2w       : (B,4,4) camera-to-world
    K         : (B,3,3) intrinsics
    depth_img : (B,H,W) z-buffer
    returns   : (B,H,W,C) uint8 images
    """
    device = points.device
    B = c2w.shape[0]
    N, C = heatmap.shape
    H, W = image_shape

    # 1) Unproject to camera coords: (B,N,3)
    pts_h = torch.cat([points, points.new_ones(N, 1)], dim=1)  # (N,4)
    w2c = torch.inverse(c2w)  # (B,4,4)
    cam_pts = w2c @ pts_h.t().unsqueeze(0)  # (B,4,N)
    cam_pts = cam_pts[:, :3, :].permute(0, 2, 1)  # (B,N,3)

    # 2) Keep only points in front of camera
    z = cam_pts[..., 2]  # (B,N)
    front_mask = z > 0

    # 3) Project to pixel coords
    proj = K @ cam_pts.permute(0, 2, 1)  # (B,3,N)
    proj = proj.permute(0, 2, 1)  # (B,N,3)
    uv = proj[..., :2] / proj[..., 2:3]  # (B,N,2)
    uv_int = uv.round().long()  # (B,N,2)

    # 4) Flatten out batch & point dims
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1)
    u_idx = uv_int[..., 0].reshape(-1)
    v_idx = uv_int[..., 1].reshape(-1)
    z_flat = z.reshape(-1)
    pid_flat = (
        torch.arange(N, device=device).unsqueeze(0).expand(B, N).reshape(-1)
    )

    # 5) Validity mask: inside image, in front, and depth‐tolerance
    valid = (
        (u_idx >= 0)
        & (u_idx < W)
        & (v_idx >= 0)
        & (v_idx < H)
        & front_mask.reshape(-1)
    )
    b_idx = b_idx[valid]
    v_idx = v_idx[valid]
    u_idx = u_idx[valid]
    z_flat = z_flat[valid]
    pid_flat = pid_flat[valid]

    d_at_pix = depth_img.reshape(B, -1)[b_idx, v_idx * W + u_idx]
    valid = (d_at_pix - z_flat).abs() < depth_tol

    b_idx = b_idx[valid]
    u_idx = u_idx[valid]
    v_idx = v_idx[valid]
    pid = pid_flat[valid]

    # 6) Gather per‐point colors
    cols = heatmap[pid]  # (M, C), float32

    # 7) Scatter‐add into (B,C,H,W) color & (B,1,H,W) weight
    color_acc = torch.zeros(B, C, H, W, device=device)
    weight_acc = torch.zeros(B, 1, H, W, device=device)

    # scatter colors per‐channel
    for c in range(C):
        color_acc[b_idx, c, v_idx, u_idx] += cols[:, c]

    # scatter weights
    ones = cols.new_ones(len(b_idx))
    weight_acc[b_idx, 0, v_idx, u_idx] += ones

    # 8) Dilate via group‐convolution
    if radius > 0:
        kernel = torch.ones(
            1, 1, 2 * radius + 1, 2 * radius + 1, device=device
        )
        # depthwise conv: reshape so each channel is its own batch
        c_flat = color_acc.view(B * C, 1, H, W)
        c_flat = F.conv2d(c_flat, kernel, padding=radius)
        color_acc = c_flat.view(B, C, H, W)
        weight_acc = F.conv2d(weight_acc, kernel, padding=radius)

    # 9) Normalize & Gaussian‐blur
    color_acc = color_acc / (weight_acc + 1e-6)
    color_acc = color_acc.cpu().numpy()
    # apply blur per channel
    out = []
    for c in range(C):
        ch = gaussian_filter(color_acc[:, c], sigma=sigma)
        out.append(ch)
    out = np.stack(out, axis=-1)  # (B,H,W,C)
    return out


def run_stage4_projection(
    data_dir: str = "/datasets/objaverse",
    subset: str = "all",
    anno_dir: str = "/datasets/objaverse/annotations",
    save_dir: str = "/datasets/objaverse/affogato_all",
    num_images: int = 25,
    image_skip: int = 2,
    target_size: int = 224,
    erode_radius: int = 2,
    smooth_sigma: float = 0.5,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    overwrite: bool = False,
):
    logger.info("==> Running stage 4: projection")
    logger.info(f"==> num_images: {num_images}, image_skip: {image_skip}")
    logger.info(f"==> anno_dir: {anno_dir}")
    logger.info(f"==> save_dir: {save_dir}")

    dataset = ObjaverseDatasetTar(
        data_dir=data_dir,
        subset=subset,
        render_dir="gobjaverse_reduced",
        load_image_only=False,
        num_images=num_images,
        image_skip=image_skip,
        rank=rank,
        world_size=world_size,
    )

    save_dir = Path(save_dir)
    filepaths = sorted(save_dir.glob("**/xyzc.npy"))
    object_ids = [f"{p.parent.parent.name}/{p.parent.name}" for p in filepaths]
    if rank is not None and world_size is not None:
        object_ids = object_ids[rank::world_size]

    print(f"==> {len(object_ids)} objects to process")
    for object_id in tqdm(object_ids):
        xyzc_path = Path(anno_dir) / object_id / "xyzc.npy"
        if not xyzc_path.exists():
            logger.warning(f"==> {object_id} does not have xyzc.npy")
            continue

        xyzc = np.load(xyzc_path)
        try:
            sample = dataset.get_by_object_id(object_id)
        except Exception as e:
            logger.warning(f"==> {object_id} failed: {e}")
            continue

        images = sample["images"]
        depths = sample["depths"]
        c2ws = sample["c2ws"]
        Ks = sample["Ks"]
        object_id = sample["object_id"]

        scale = target_size / depths[0].shape[1]
        depth_resized = [
            cv2.resize(depth, (target_size, target_size)) for depth in depths
        ]
        Ks_resized = []
        for K in Ks:
            K_resized = K.copy()
            K_resized[0, 0] *= scale
            K_resized[1, 1] *= scale
            K_resized[0, 2] *= scale
            K_resized[1, 2] *= scale
            Ks_resized.append(K_resized)

        xyzc = np.load(Path(anno_dir) / object_id / "xyzc.npy")
        projections_all = render_heatmap_batch_torch(
            points=torch.from_numpy(xyzc[:, :3]).float().cuda(),
            heatmap=torch.from_numpy(xyzc[:, 3:]).float().cuda(),
            c2w=torch.from_numpy(np.stack(c2ws)).float().cuda(),
            K=torch.from_numpy(np.stack(Ks_resized)).float().cuda(),
            depth_img=torch.from_numpy(np.stack(depth_resized)).float().cuda(),
            image_shape=(target_size, target_size),
            sigma=smooth_sigma,
            radius=erode_radius,
        )
        most_visible_image_indices = projections_all.sum(axis=(1, 2)).argmax(
            axis=0
        )
        indices_offset = np.random.randint(
            1, 4, size=most_visible_image_indices.shape
        )
        indices_offset = (
            np.random.choice([-1, 1], size=most_visible_image_indices.shape)
            * indices_offset
        )
        local_visible_image_indices = (
            most_visible_image_indices + indices_offset
        ) % len(images)
        # save results
        image_indices = set(most_visible_image_indices) | set(
            local_visible_image_indices
        )
        for i in image_indices:
            img = images[i].resize((target_size, target_size))
            img.save(Path(save_dir) / object_id / f"color_{i}.png")
            mask = ((depth_resized[i] > 0) * 255).astype(np.uint8)
            cv2.imwrite(
                Path(save_dir) / object_id / f"depth_mask_{i}.png", mask
            )

        for query_idx, (idx0, idx1) in enumerate(
            zip(local_visible_image_indices, most_visible_image_indices)
        ):
            projection0 = projections_all[idx0, :, :, query_idx]
            projection1 = projections_all[idx1, :, :, query_idx]

            def normalize_projection(projection):
                max_value = projection.max()
                projection = projection / (max_value + 1e-6)
                projection = np.power(projection, 2)
                projection = (projection * 255).clip(0, 255).astype(np.uint8)
                return projection

            projection0 = normalize_projection(projection0)
            projection1 = normalize_projection(projection1)

            cv2.imwrite(
                Path(save_dir)
                / object_id
                / f"projection_{query_idx}_main_{idx0}.png",
                projection0,
            )
            cv2.imwrite(
                Path(save_dir)
                / object_id
                / f"projection_{query_idx}_local_{idx1}.png",
                projection1,
            )
        logger.info(f"==> {object_id} done")


if __name__ == "__main__":
    import fire

    fire.Fire(run_stage4_projection)
