import matplotlib.pyplot as plt
import numpy as np


def array_to_jet_colormap(
    array: np.ndarray, cmap_str: str = "jet"
) -> np.ndarray:
    assert np.all((array >= 0) & (array <= 1)), (
        "Array values must be in [0, 1] range."
    )

    cmap = plt.get_cmap(cmap_str)
    rgb = cmap(array)[..., :3]  # Drop the alpha channel
    return rgb


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
            for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, ax, marker_size=375):
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_heatmap(logits, ax):
    # Apply sigmoid normalization to logits
    normalized_logits = 1 / (1 + np.exp(-logits))

    # Apply jet colormap
    colored_logits = array_to_jet_colormap(normalized_logits)

    # Display the colored logits with transparency
    ax.imshow(colored_logits, alpha=0.7)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
        )
    )
