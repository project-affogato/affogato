import numpy as np
import trimesh


def normalize_mesh(mesh: trimesh.Trimesh):
    bounds = mesh.bounding_box.bounds
    bounds_min = bounds[0]
    bounds_max = bounds[1]
    scale_tmp = max(
        (bounds_max[0] - bounds_min[0]), (bounds_max[1] - bounds_min[1])
    )
    scale_tmp = max((bounds_max[2] - bounds_min[2]), scale_tmp)
    scale_tmp = 0.9 / scale_tmp

    offset = -(bounds_max + bounds_min) / 2
    return scale_tmp, offset


def sample_points_on_mesh(
    mesh: trimesh.Trimesh, num_points: int, sample_color: bool = False
):
    """
    Sample points on a mesh.
    """
    output = trimesh.sample.sample_surface(
        mesh, num_points, sample_color=sample_color
    )
    if sample_color:
        points, _, colors = output
    else:
        points, _ = output
        colors = None
    return points, colors


def unproject_depth(depth_map, c2w, K, depth_threshold=0.0, blur_sigma=1.0):
    """
    Unproject a depth map to point cloud using camera parameters.
    """
    # Unproject depth map to point cloud using camera matrix and intrinsic matrix
    from scipy.ndimage import gaussian_filter

    depth_blur = gaussian_filter(depth_map, sigma=blur_sigma)

    cond_cam_dis = np.linalg.norm(c2w[:3, 3])
    near = 0.867
    near_distance = cond_cam_dis - near
    depth_blur[depth_blur < near_distance] = 0
    depth_map = depth_blur

    h, w = depth_map.shape
    # Create pixel coordinate grid
    y, x = np.mgrid[0:h, 0:w]

    # Flatten coordinates and depth
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()
    # Filter out invalid depth values
    valid_mask = ~np.isnan(z) & (z > depth_threshold)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    # Create homogeneous pixel coordinates
    pixels = np.stack([x, y, np.ones_like(x)], axis=0)

    # Unproject to camera space
    rays = (np.linalg.inv(K) @ pixels).T
    points_camera = rays * z[:, None]

    # Convert to homogeneous coordinates
    points_camera_homogeneous = np.concatenate(
        [points_camera, np.ones((points_camera.shape[0], 1))], axis=1
    )

    # Transform to world coordinates
    points_world = (c2w @ points_camera_homogeneous.T).T[:, :3]
    return points_world


# feat_2d.shape = (C, H, W)
# mapping = (Y, X) (H, W)
# feat_2d[:, mapping[0], mapping[1]]
class PointCloudToImageMapper(object):
    def __init__(
        self,
        image_dim,
        visibility_threshold=0.25,
        cut_bound=0,
        intrinsics=None,
    ):
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(
        self, camera_to_world, coords, depth=None, intrinsic=None
    ):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None:  # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate(
            [coords, np.ones([coords.shape[0], 1])], axis=1
        ).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                np.abs(
                    depth[pi[1][inside_mask], pi[0][inside_mask]]
                    - p[2][inside_mask]
                )
                <= self.vis_thres * depth_cur
            )

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T
