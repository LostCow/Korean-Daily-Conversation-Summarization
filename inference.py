import os
import json

import fire
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from korean_dcs_2024.trainer.dataset import CustomDataset


def inference(
    model_name: str,
    data_path: str,
    output_path: str,
    cache_dir: str = "/data/.cache",
    do_submit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map="auto"
    )

    model.eval()

    dataset = CustomDataset(
        data_path,
        tokenizer=tokenizer,
        do_tapt=False,
        one_shot=True,
        oneshot_idx=0,
    )

    with open(data_path, "r", encoding="utf-8") as f:
        origin_data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        with open(output_path, "w", encoding="utf-8") as f:
            for idx in tqdm(range(len(dataset))):
                inp = dataset.source[idx]
                outputs = model.generate(
                    inp.to("cuda").unsqueeze(0),
                    max_new_tokens=1024,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=1.0,
                    use_cache=True,
                )

                if do_submit:
                    origin_data[idx]["output"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
                else:
                    output = origin_data[idx]
                    output["predict"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")

    if do_submit:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(origin_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    fire.Fire(inference)