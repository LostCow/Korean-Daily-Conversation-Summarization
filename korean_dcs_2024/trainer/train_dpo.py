import os
import sys

import json
import fire
import torch
from datasets import Dataset

from peft import (
    LoraConfig,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig


def train(
    dataset_path: str,
    base_model: str = "rtzr/ko-gemma-2-9b-it", 
    output_dir: str = "gemma-dpo",
    num_epochs: int = 1,
    learning_rate: float = 5e-7,
    cutoff_len: int = 4096,
    lr_scheduler: str = "linear",
    warmup_ratio: float = 0.1, 
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    training_args = DPOConfig(
        beta=0.08,
        num_train_epochs= num_epochs,
        deepspeed="korean_dcs_2024/trainer/ds_config_zero_2.json",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_only_model=True,
        dataloader_num_workers=4,
        learning_rate=learning_rate,
        output_dir=output_dir,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        optim='adamw_torch_fused',
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=33,
        save_steps=33,
        fp16=True,
        tf32=True,
        remove_unused_columns=False,
        report_to="none",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        use_cache=False, 
        attn_implementation="eager",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(type(model))
    print(model)
    print("length of tokenizer:",len(tokenizer))


    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"


    with open(dataset_path, "r") as f:
        train_data = json.load(f)


    dataset = Dataset.from_dict(train_data)
    dataset = dataset.train_test_split(test_size=0.01)

    dataset = dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= cutoff_len
        and len(x["prompt"]) + len(x["rejected"]) <= cutoff_len
    )
    train_dataset = dataset["train"].shuffle()
    dev_dataset = dataset["test"]
    print(train_dataset['prompt'][0])
    print(f"chosen: {train_dataset['chosen'][0]}")
    print(f"rejected: {train_dataset['rejected'][0]}")
    

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model = None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=cutoff_len,
        max_prompt_length=3450,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # train
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    # save
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)