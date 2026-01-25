from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_utils import PreTrainedModel

from .config import FLORENCE_MODEL_NAME

if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False
if not hasattr(torch.nn.Module, "_supports_sdpa"):
    torch.nn.Module._supports_sdpa = False


class FlorenceCaptioner:
    def __init__(self, model_name: str = FLORENCE_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = self._load_model(model_name)
        self.model.config.use_cache = False
        self.model.eval()

    def caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        prompt = prompt or "<MORE_DETAILED_CAPTION>"
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=1,
                use_cache=False,
            )

        decoded = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return decoded.strip()

    def _load_model(self, model_name: str):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        return model.to(self.device)
