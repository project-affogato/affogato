# %%
import sys

sys.path.append("/workspace")

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jupythree.pointcloud import pointcloud

from src.dataset.objaverse.dataset import ObjaverseDatasetTar
from src.utils.vis import array_to_jet_colormap

# %%
data_dir = Path("/datasets/objaverse")
subset = "Furnitures"
dataset = ObjaverseDatasetTar(
    data_dir,
    subset,
    render_dir="gobjaverse_reduced",
    load_image_only=True,
)

anno_dir = data_dir / "annotations"
object_ids = dataset.object_ids

# filter object_ids that have xyzc.npy
object_ids = sorted(
    [
        obj_id
        for obj_id in object_ids
        if os.path.exists(os.path.join(anno_dir, obj_id, "xyzc.npy"))
    ]
)
print(len(object_ids))

# %% [markdown]
# ## Visualize Queries

# %%
idx = 120
object_id = object_ids[idx]
object_dir = os.path.join(anno_dir, object_id)

xyzc = np.load(os.path.join(object_dir, "xyzc.npy"))
query_data = json.load(open(os.path.join(object_dir, "queries.json")))[0]
class_name, queries = query_data["class_name"], query_data["queries"]

images = dataset.get_by_object_id(object_id)["images"][::5][:5]

for q in queries:
    print(q)
fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
for ax, image in zip(axes, images):
    ax.imshow(image)
    ax.axis("off")
plt.show()

# %% [markdown]
# ## Visualize 2D Points (molmo)

# %%
num_images = 25
point_file = "points.npy"
skip = re.search(r"_skip(\d+)\.npy$", point_file)
if skip is not None:
    skip = int(skip.group(1))
else:
    skip = 1

idx = 120
object_id = object_ids[idx]
object_dir = os.path.join(anno_dir, object_id)

query_data = json.load(open(os.path.join(object_dir, "queries.json")))[0]
class_name, queries = query_data["class_name"], query_data["queries"]
points_all = np.load(os.path.join(object_dir, point_file))
points_all = np.transpose(
    points_all, (1, 0, 2)
)  # (num_images, num_queries, 2)
images = dataset.get_by_object_id(object_id)["images"][:num_images][::skip]
assert len(images) == points_all.shape[0]
assert len(queries) == points_all.shape[1]

for q in queries:
    print(q)

fig, axes = plt.subplots(
    len(images), points_all.shape[1], figsize=(5, len(images))
)
for ax, image, points in zip(axes, images, points_all):
    for a in ax:
        a.axis("off")
        a.imshow(image)
    for i in range(points.shape[0]):
        ax[i].scatter(
            points[i, 0], points[i, 1], color="blue", s=20, marker="*"
        )
plt.show()
# %% [markdown]
# ## Visualize Pointclouds

# %%
idx = 20
object_id = object_ids[idx]
object_dir = os.path.join(anno_dir, object_id)

xyzc = np.load(os.path.join(object_dir, "xyzc.npy"))
query_data = json.load(open(os.path.join(object_dir, "queries.json")))[0]
class_name, queries = query_data["class_name"], query_data["queries"]

query_idx = 0
xyz = xyzc[:, :3]
heatmaps = xyzc[:, 3:]
heatmap = heatmaps[:, query_idx]
color = array_to_jet_colormap(heatmap)

print(queries[query_idx])
pc = pointcloud(xyz, color)
pc.show(width=800)
