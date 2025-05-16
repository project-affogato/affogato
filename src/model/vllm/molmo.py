import re
from dataclasses import asdict
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest


def extract_points(
    molmo_output: str, image_wh: Tuple[int, int], return_all: bool = False
) -> np.array:
    """
    Obtained from https://huggingface.co/allenai/Molmo-7B-O-0924/discussions/1
    """
    image_w, image_h = image_wh
    all_points = []
    for match in re.finditer(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
        molmo_output,
    ):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            points = all_points.append(point)
    if len(all_points) > 0:
        points = np.stack(all_points, axis=0)
        if not return_all:
            points = points[:1]  # pick the first point
    else:
        points = np.array([[image_w / 2, image_h / 2]])  # fallback to center
    return points


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


class Molmo:
    def __init__(
        self,
        model_id: str = "allenai/Molmo-7B-D-0924",
        use_tqdm: bool = False,
        max_new_tokens: int = 64,
        seed: int = 42,
    ):
        self.model_id = model_id
        self.use_tqdm = use_tqdm
        self.engine_args = EngineArgs(
            model=model_id,
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
        )

        llm_args = asdict(self.engine_args) | {"seed": seed}
        self.model = LLM(**llm_args)
        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=max_new_tokens,
            stop_token_ids=None,
        )

    def process(
        self,
        images: List[Image.Image],
        query: Union[str, List[str]],
        *args,
        **kwargs,
    ) -> str:
        if isinstance(query, str):
            query = [query]

        prompt = "<|im_start|>user\n<image>\n{query}<|im_end|>\n<|im_start|>assistant\n"
        image_wh = images[0].size
        num_queries = len(query)
        num_images = len(images)

        messages = [
            {
                "prompt": prompt.format(query=q),
                "multi_modal_data": {"image": img},
            }
            for q in query
            for img in images
        ]
        response = self.model.generate(
            messages,
            sampling_params=self.sampling_params,
            use_tqdm=self.use_tqdm,
        )
        generated_texts = [o.outputs[0].text for o in response]
        points = np.concatenate(
            [extract_points(text, image_wh) for text in generated_texts]
        ).reshape((num_queries, num_images, -1))
        return points


if __name__ == "__main__":
    molmo = Molmo()
    points = molmo.process(
        images=[
            Image.open(
                f"demo/16341/campos_512_v4/{i:05d}/{i:05d}.png"
            ).convert("RGB")
            for i in range(5)
        ],
        query=["Point to the part of the chair that human sit on."] * 2,
    )
    import pdb

    pdb.set_trace()
