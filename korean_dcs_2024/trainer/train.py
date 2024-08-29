import logging
import os
from dataclasses import asdict

import torch
from datasets import Dataset
import datasets
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from arguments import (
    DatasetsArgs,
    LoraArgs,
    ModelArgs,
    QuantizationArgs,
    TrainingArgs,
)
from dataset import (
    CustomDataset,
    DataCollatorForSupervisedDataset,
)


def train(
    model_args: ModelArgs,
    data_args: DatasetsArgs,
    training_args: TrainingArgs,
):
    bnb_config = None
    if training_args.quantization:
        quant_config = QuantizationArgs()
        bnb_config = quant_config.create_bnb_config(training_args.quantization)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": True,
        "max_position_embeddings": model_args.model_max_length,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
        attn_implementation="eager" if "gemma" in model_args.model_name_or_path else "flash_attention_2",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    # set gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_args.model_max_length,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # resize token embeddings to model_max_length
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.use_peft:
        peft_config = LoraConfig(**asdict(LoraArgs()))
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    d_train = CustomDataset(
        data_args.train_dataset_path,
        tokenizer,
        training_args.do_tapt,
        one_shot=training_args.one_shot,
        use_custom_prompt=training_args.use_custom_prompt,
        multitask=training_args.multitask,
    )
    train_dataset = Dataset.from_dict(
        {
            "input_ids": d_train.inp,
            "labels": d_train.label,
            "attention_mask": d_train.attention_mask,
        }
    )
    train_dataset.set_format(
        type="torch", columns=["input_ids", "labels", "attention_mask"]
    )

    print(f"Train dataset: {len(train_dataset)}")
    print(f'input: {tokenizer.decode(train_dataset[0]["input_ids"])}')
    print(f'label: {tokenizer.decode(torch.where(train_dataset[0]["labels"] == -100, tokenizer.pad_token_id, train_dataset[0]["labels"]))}')

    d_eval = CustomDataset(
        data_args.validation_dataset_path,
        tokenizer,
        training_args.do_tapt,
        one_shot=training_args.one_shot,
        use_custom_prompt=training_args.use_custom_prompt,
        multitask=training_args.multitask,
    )
    eval_dataset = Dataset.from_dict(
        {
            "input_ids": d_eval.inp,
            "labels": d_eval.label,
            "attention_mask": d_eval.attention_mask,
        }
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "labels", "attention_mask"]
    )

    if training_args.with_train_dev:
        train_dataset = datasets.concatenate_datasets([train_dataset, eval_dataset])

    generation_config = model.generation_config
    generation_config.max_new_tokens = data_args.eval_generation_max_new_tokens
    training_args.generation_config = generation_config

    data_collator_with_padding = DataCollatorForSupervisedDataset(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator_with_padding,
    )

    # set checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    # train
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # log and save train metrics
    metrics = train_result.metrics

    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # push to hub or create model card
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-generation",
    }
    kwargs["dataset_tags"] = data_args.train_dataset_path

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_dotenv()

    parser = HfArgumentParser((ModelArgs, DatasetsArgs, TrainingArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )

    # set seed, default=42
    set_seed(training_args.seed)

    # set wandb
    if is_main_process(training_args.local_rank) and "wandb" in training_args.report_to:
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            name=training_args.run_name,
            group=training_args.wandb_group,
        )

    train(model_args=model_args, data_args=data_args, training_args=training_args)
