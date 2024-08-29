from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
from transformers import BitsAndBytesConfig, Seq2SeqTrainingArguments


@dataclass
class LoraArgs:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]
    )
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False


@dataclass
class QuantizationArgs:
    quant_type: str = "nf4"  # "fp4" or "nf4"
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = False
    quant_storage: torch.dtype = torch.bfloat16

    def create_bnb_config(self, quantization: str) -> BitsAndBytesConfig:
        if quantization not in {"4bit", "8bit"}:
            raise ValueError("quantization must be either '4bit' or '8bit'")

        if quantization == "4bit":
            config_params = {
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_compute_dtype": self.compute_dtype,
                "bnb_4bit_use_double_quant": self.use_double_quant,
                "bnb_4bit_quant_storage": self.quant_storage,
            }

            return BitsAndBytesConfig(load_in_4bit=True, **config_params)
        else:
            return BitsAndBytesConfig(load_in_8bit=True)


@dataclass
class ModelArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/data/yohan/ko-gemma-2-9b-it",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_max_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": "The maximum length the generated tokens can have. Corresponds to the length of the input prompt + \
                `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in \
                the prompt."
        },
    )
    cache_dir: Optional[str] = field(
        default="/data/.cache",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DatasetsArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_path: str = field(
        default="korean_dcs_2024/Korean_DCS_2024_일상대화요약_데이터/일상대화요약_train.json",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    validation_dataset_path: str = field(
        default="korean_dcs_2024/Korean_DCS_2024_일상대화요약_데이터/일상대화요약_dev.json",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    eval_generation_max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum number of new tokens to generate when using `eval_generate`."
        },
    )


@dataclass
class TrainingArgs(Seq2SeqTrainingArguments):
    wandb_project: Optional[str] = field(
        default="", metadata={"help": "wandb project name for logging"}
    )
    wandb_group: Optional[str] = field(
        default="", metadata={"help": "wandb group name for logging"}
    )

    quantization: Optional[Literal["8bit", "4bit"]] = field(
        default="", metadata={"help": "Whether to use quantization"}
    )
    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT"})

    do_tapt: bool = field(default=False, metadata={"help": "Whether to do TAPT"})

    use_custom_prompt: bool = field(
        default=False, metadata={"help": "Whether to use custom prompt"}
    )
    multitask: bool = field(default=False, metadata={"help": "Whether to use multitask"})
    with_train_dev: bool = field(
        default=False, metadata={"help": "Whether to use train dev data"}
    )
    one_shot: bool = field(default=False, metadata={"help": "Whether to use one shot"})