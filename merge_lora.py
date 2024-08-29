import os
import fire

from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    PeftModel,
)

from glob import glob

def merge_lora(
    peft_model_name_or_path: str = "ko-gemma-2-9b-it-tapt",
    base_model_name_or_path: str = "/data/yohan/ko-gemma-9b",
    cache_dir: str = "/data/.cache",
    hub_model_id: str = None,
    push_to_hub: bool = False,
):
    targets = glob(f"{peft_model_name_or_path}/checkpoint*")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    for target in targets:
        peft_model = PeftModel.from_pretrained(
            base_model,
            target,
        )

        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(target, "merged"))

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            cache_dir=cache_dir,
        )

        tokenizer.save_pretrained(os.path.join(target, "merged"))

        if push_to_hub and hub_model_id:
            token = os.getenv("HUGGINGFACE_AUTH_TOKEN")
            merged_model.push_to_hub(hub_model_id, private=True, token=token)
            tokenizer.push_to_hub(hub_model_id, private=True, token=token)



if __name__ == '__main__':
    load_dotenv()
    fire.Fire(merge_lora)