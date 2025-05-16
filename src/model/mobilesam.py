from typing import Literal

import huggingface_hub
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry
from PIL.Image import Image


def download_checkpoint():
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id="dhkim2810/MobileSAM",
        filename="mobile_sam.pt",
    )
    return checkpoint_path


class MobileSAM:
    def __init__(
        self,
        mask_selection_mode: Literal[
            "highest_score", "smallest_mask", "random"
        ] = "smallest_mask",
    ):
        checkpoint_path = download_checkpoint()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_type = "vit_t"
        mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        mobile_sam = mobile_sam.to(device=self.device)
        mobile_sam.eval()
        self.model = SamPredictor(mobile_sam)
        assert mask_selection_mode in [
            "highest_score",
            "smallest_mask",
            "random",
        ], "Invalid mask selection mode"
        self.mask_selection_mode = mask_selection_mode
        self.native_resolution = 1024

    @torch.inference_mode()
    def _inference_sam(self, image, points):
        self.model.set_image(image)
        point_coords = torch.from_numpy(points[:, np.newaxis, :]).to(
            self.device
        )
        point_labels = torch.ones(len(points)).unsqueeze(1).to(self.device)
        logits, scores, _ = self.model.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=True,
        )
        logits = logits.cpu().numpy()
        scores = scores.cpu().numpy()
        masks = logits > 0
        return masks, scores, logits

    def process(self, image: np.ndarray, points: np.ndarray):
        if isinstance(image, Image):
            image = np.array(image)

        h, w = image.shape[:2]
        scale = self.native_resolution / max(h, w)
        points_scaled = points * scale
        masks, scores, logits = self._inference_sam(image, points_scaled)
        arange = np.arange(len(masks))
        if self.mask_selection_mode == "highest_score":
            argmax = np.argmax(scores, axis=-1)
            mask = masks[arange, argmax]
            score = scores[arange, argmax]
            logit = logits[arange, argmax]
        elif self.mask_selection_mode == "smallest_mask":
            mask_area = masks.sum(axis=(-2, -1))
            argmin = np.argmin(mask_area, axis=-1)
            mask = masks[arange, argmin]
            score = scores[arange, argmin]
            logit = logits[arange, argmin]
        elif self.mask_selection_mode == "random":
            idx = np.random.choice(masks.shape[0])
            mask = masks[arange, idx]
            score = scores[arange, idx]
            logit = logits[arange, idx]
        return mask, score, logit


if __name__ == "__main__":
    import cv2

    images = [
        cv2.imread(f"demo/16341/campos_512_v4/{i:05d}/{i:05d}.png")
        for i in range(1, 5)
    ]
    model = MobileSAM()
    for image in images:
        masks, scores, logits = model.process(image, np.array([[100, 100]]))
        print(masks)
        print(scores)
        print(logits)
