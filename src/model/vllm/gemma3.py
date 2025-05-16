import os
from dataclasses import asdict

import yaml
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor
from vllm import LLM, EngineArgs, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from src.utils.logging import get_logger
from src.utils.misc import parse_json_string

logger = get_logger(__name__)


class AffordanceQuery(BaseModel):
    class_name: str
    queries: list[str]


class Gemma3:
    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        max_model_len: int = 8192,
        max_num_seqs: int = 2,
        max_num_images: int = 5,
        max_new_tokens: int = 256,
        num_queries: int = 5,
        seed: int = 42,
        use_tqdm: bool = True,
        prompt_file: str = "gemma3.yaml",
    ):
        self.model_id = model_id
        self.num_queries = num_queries
        self.use_tqdm = use_tqdm
        self.engine_args = EngineArgs(
            model=model_id,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            dtype="bfloat16",
            limit_mm_per_prompt={
                "image": max_num_images,
                "video": 0,
                "audio": 0,
            },
        )
        llm_args = asdict(self.engine_args) | {"seed": seed}
        self.model = LLM(**llm_args)
        self.processor = AutoProcessor.from_pretrained(model_id)
        # guided decoding
        json_schema = AffordanceQuery.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=1.0,
            max_tokens=max_new_tokens,
            stop_token_ids=None,
            guided_decoding=guided_decoding_params,
        )
        with open(os.path.join("src/model/prompt", prompt_file), "r") as f:
            self.prompt_template = yaml.safe_load(f)

        logger.info(
            f"{self.__class__.__name__} initialized with prompt template: {prompt_file}"
        )

    def process(self, images: list[str]) -> str:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template["SYSTEM_PROMPT"],
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": image_path}
                        for image_path in images
                    ],
                    {
                        "type": "text",
                        "text": self.prompt_template["USER_PROMPT"].format(
                            n_views=len(images),
                            n_queries=self.num_queries,
                        ),
                    },
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        payload = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": images,
            },
        }
        response = self.model.generate(
            payload,
            sampling_params=self.sampling_params,
            use_tqdm=self.use_tqdm,
        )
        generated_texts = [o.outputs[0].text for o in response]
        parsed_texts = [parse_json_string(text) for text in generated_texts]
        return parsed_texts


if __name__ == "__main__":
    gemma3 = Gemma3()
    print(
        gemma3.process(
            [
                Image.open(
                    f"demo/16341/campos_512_v4/{i:05d}/{i:05d}.png"
                ).convert("RGB")
                for i in range(5)
            ],
        )
    )
