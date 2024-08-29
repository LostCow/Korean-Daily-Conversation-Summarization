import json
from itertools import chain

import datasets
import torch
from torch.utils.data import Dataset
import pandas as pd


FEW_SHOT_EXAMPLE_1 = """[Example 1]
You are a helpful AI assistant. Please answer the user's questions kindly.
당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.

[Conversation]
SD2000001: 저는 여행 다니는 것을 굉장히 좋아하는데요. 그래가지고 스페인이나 뭐 영국 유럽 아니면 국내에서도 뭐 강릉이나 전주 같은 데를 많이 다녔는데
SD2000001: 혹시 여행 다니는 거 좋아하시나요?
SD2000002: 저 여행 다니는 거 되게 좋아해서 대학교 내내 여행을 엄청 많이 다녔었는데요.
SD2000002: 제가 고등학교 때는 여행에 대해 흥미가 없었는데 그게 좀 아버지가 짠대로 패키지처럼 여행을 다녀서 그런 것 같아요.
SD2000002: 그래서 대학교 간 이후로는 해외여행을 되게 많이 갔었는데 그중에서 제일 기 좋았던 거는 스페인이랑 포르투갈이었거든요.
SD2000002: 어~ 혹시 포르투갈이나 스페인 유럽 쪽 다녀오신 적 있으신가요?
SD2000001: 어~ 네. 저도 우연히 스페인과 포르투갈을 다녀왔었었습니다.
SD2000001: 어~ 저는 스페인 중에서도 마드리드에 근교에 있었던 톨레도라는 지역이 굉장히 좋았는데요. 그 톨레도에서 특히 기억에 남았던 거는 거기에 대성당이 있는데 그 성당이 엄청 화려하더라고요. 그래서 거기를 꾸며논 거를 보면은 금을 엄청 많이 사용해가지고 되게 빤짝빤짝하고 좀 성당은 보통 좀 소박하다라는 인식이 있었는데 아~ 이렇게 화려한 성당도 있구나라는 거를 새롭게 알게 됐었습니다.
SD2000001: 어~ 또 톨레도에 지역 음식도 같이 먹었었는데 아~ 이름은 지금 잘 생각이 나지는 않지만 굉장히 달달했던 그런 디저트 종류였는데 그~ 디저트도 먹고 그다음에 천천히 걸어 다니면서 주변 풍경도 보고 근교 여행만의 약간 소박한 맛이 있었다고 생각을 합니다.
SD2000001: 어~ 또 물론 마드리드도 굉장히 좋았는데 유럽 여행을 많이 가셨다고 해서 혹시 톨레도도 가본 적이 있나요?
SD2000002: 아~ 제가 톨레도도 다녀왔는데 저는 이제 여행 일정을 길게 잡아서 톨레도는 하루를 봤는데 도 그렇게 너무 더웠기 때문에 많이 보진 못한 것 같아요.
SD2000002: 그때는 버스 관광버스를 타고 계속 돌아다니면서 이제 내리는 데마다 관광을 할 수 있는 버스를 탔는데요. 그 버스를 타고 전체를 다 내려서 보려고 했지만 날씨가 너무 더워서 금방 금방 이제 xx 장소로 넘어갔던 것 같 같습니다.
SD2000002: 거기는 이제 고대 도시라고 해서 사람들이 많이 추천한 거에 비해서는 저는 하루를 잡기에는 조금 부족한 여행지라는 생각이 들었고
SD2000002: 오히려 광장에서 쇼핑을 했던 게 더 기억에 남습니다.

[Question]
위 해외여행 주제에 대한 대화를 요약해주세요.

[Answer]
이 대화에서 화자들은 좋았던 여행지와 기억나는 주요 명소에 대해 이야기했습니다. SD2000001은 여행을 좋아하여 국내, 해외 여행을 많이 다녔다고 말했습니다. 특히 기억에 남는 여행지로 스페인 마드리드의 톨레도를 소개했습니다. 그 중 화려하게 꾸며진 대성당과 디저트가 인상적이었다고 이야기했습니다. SD2000002는 대학교에 진학한 후 해외여행을 자주 다녔고, 스페인과 포루투갈이 가장 기억에 남는 여행지라고 말했습니다. 그리고 톨레도도 다녀왔지만 날씨가 더워서 제대로 구경하지 못했다는 경험을 이야기했습니다."""


class CustomDataset(Dataset):
    def __init__(
        self,
        fname,
        tokenizer,
        do_tapt: bool,
        one_shot: bool = False,
        dynamic_fewshot: bool = False,
        fewshot_block_size: int = 8192,
        use_custom_prompt: bool = False,
        multitask: bool = False,
        oneshot_idx: int = 0,
    ):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []
        self.attention_mask = []
        self.source = []
        self.chat_completion_messages = []
        self.chat_completion_messages_with_generation_prompt = []
        self.tokenizer = tokenizer
        self.fewshot_block_size = fewshot_block_size

        self.fewshot_df = pd.read_csv("few_shot.csv")

        with open(fname, "r") as f:
            data = json.load(f)

        def add_multitask():
            PROMPT = """You are a helpful AI assistant. Please answer the user's questions kindly.
당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."""
            for example in data:
                inp = example["input"]

                chat = ["[Conversation]"]
                for cvt in inp["conversation"]:
                    speaker = cvt["speaker"]
                    utterance = cvt["utterance"]
                    chat.append(f"{speaker}: {utterance}")
                chat = "\n".join(chat)

                question = (
                    f"[Question]\n위 대화의 주제를 추출해주세요."
                )

                chat = chat + "\n\n" + question

                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chat},
                ]

                self.chat_completion_messages.append(message)

                source = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                self.source.append(source[0])

                target = ', '.join(inp['subject_keyword'])
                if target != "":
                    target += tokenizer.eos_token

                target = tokenizer(
                    target,
                    return_attention_mask=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                target["input_ids"] = target["input_ids"].type(torch.int64)

                input_ids = torch.concat((source[0], target["input_ids"][0]))
                labels = torch.concat(
                    (
                        torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]),
                        target["input_ids"][0],
                    )
                )
                self.inp.append(torch.LongTensor(input_ids))
                self.label.append(torch.LongTensor(labels))
                self.attention_mask.append(torch.ones_like(input_ids))



        def make_chat(inp, do_tapt: bool = False):
            chat = ["[Conversation]"]
            for cvt in inp["conversation"]:
                speaker = cvt["speaker"]
                utterance = cvt["utterance"]
                chat.append(f"{speaker}: {utterance}")
            chat = "\n".join(chat)
            if do_tapt:
                return chat

            question = (
                f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 요약해주세요."
            )
            chat = chat + "\n\n" + question

            return chat

        if do_tapt:
            for example in data:
                _input = example["input"]

                PROMPT = f"""You are a helpful AI assistant. Please answer the user's questions kindly.
당신은 유능한 AI 어시스턴트 입니다. 다음은 {', '.join(_input['subject_keyword'])} 주제에 대한 대화입니다."""

                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": make_chat(_input, do_tapt)},
                ]

                source = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )[0]

                labels = source.clone()

                self.inp.append(source)
                self.label.append(labels)
                self.attention_mask.append(torch.ones_like(source))
        elif use_custom_prompt:
            for example in data:
                conversation = ""
                _input = example["input"]
                for cvt in _input["conversation"]:
                    speaker = cvt["speaker"]
                    utterance = cvt["utterance"]
                    conversation += f"화자 {speaker}: {utterance}\n"

                PROMPT = f"""### Instruction
당신은 유능한 AI 요약 어시스턴트입니다. 아래 대화를 바탕으로 친절하고 객관적인 요약문을 작성하세요.
대화의 주요 주제와 핵심 발언을 중심으로 요약하며, 불필요한 세부사항은 생략합니다.
요약문은 친절한 어조로 작성하되, 논리적이고 일관된 흐름을 유지하면서 구체적인 정보를 빠짐없이 포함하도록 합니다.

### Conversation
{conversation}

### User
위 {', '.join(_input['subject_keyword'])} 주제에 대한 대화를 바탕으로 요약문을 작성하세요.

### Assistant
"""

                self.chat_completion_messages.append([{"role": "user", "content": PROMPT}])

                source = tokenizer(
                    PROMPT,
                    return_attention_mask=False,
                    return_tensors="pt",
                )["input_ids"]
                source = source.type(torch.int64)

                target = example["output"]
                if target != "":
                    target += tokenizer.eos_token

                target = tokenizer(
                    target,
                    return_attention_mask=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"]
                target = target.type(torch.int64)

                input_ids = torch.concat((source[0], target[0]))
                labels = torch.concat(
                    (
                        torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]),
                        target[0],
                    )
                )

                self.inp.append(torch.LongTensor(input_ids))
                self.label.append(torch.LongTensor(labels))
                self.attention_mask.append(torch.ones_like(input_ids))
        else:
            for example in data:
                PROMPT = """You are a helpful AI assistant. Please answer the user's questions kindly.
당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."""

                chat = make_chat(example["input"])
                if one_shot:
                    if oneshot_idx != 0:
                        oneshot_example = self.fewshot_df["text"].tolist()[oneshot_idx]
                        message = [
                            {"role": "system", "content": oneshot_example + "\n\n---\n\n" + PROMPT},
                            {"role": "user", "content": chat},
                        ]
                    else:
                        message = [
                            {"role": "system", "content": FEW_SHOT_EXAMPLE_1 + "\n\n---\n\n" + PROMPT},
                            {"role": "user", "content": chat},
                        ]
                elif dynamic_fewshot:
                    message = [
                        {
                            "role": "system", "content": self.dynamic_fewshot_insertion(
                                tokenizer.apply_chat_template(
                                    [{"role": "system", "content": PROMPT},
                                     {"role": "user", "content": chat}],
                                     tokenize=False,
                                     add_generation_prompt=True,
                                ), self.fewshot_block_size
                            ) + PROMPT
                        },
                        {"role": "user", "content": chat},
                    ]
                else:
                    message = [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": chat},
                    ]

                self.chat_completion_messages.append(message)

                source = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                self.chat_completion_messages_with_generation_prompt.append(
                    tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

                self.source.append(source[0])

                target = example["output"]
                if target != "":
                    target += tokenizer.eos_token

                target = tokenizer(
                    target,
                    return_attention_mask=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                target["input_ids"] = target["input_ids"].type(torch.int64)

                input_ids = torch.concat((source[0], target["input_ids"][0]))
                labels = torch.concat(
                    (
                        torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]),
                        target["input_ids"][0],
                    )
                )
                self.inp.append(torch.LongTensor(input_ids))
                self.label.append(torch.LongTensor(labels))
                self.attention_mask.append(torch.ones_like(input_ids))

        if multitask:
            add_multitask()


    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]
    
    def dynamic_fewshot_insertion(
        self,
        initial_string: str,
        fewshot_block_size: int,
    ) -> str:
        initial_tokens = self.tokenizer.encode(initial_string)
        current_length = len(initial_tokens)

        insert_fewshot_string = []
        for _, row in self.fewshot_df.iterrows():
            example_text = row['text']
            example_tokens = self.tokenizer.encode(example_text)
            example_length = len(example_tokens)
            
            if current_length + example_length > fewshot_block_size:
                break

            current_length += example_length

            insert_fewshot_string.append(example_text)

        output_string = ""
        for i, insert_example_text in enumerate(insert_fewshot_string):
            output_string += f"[Example {i + 1}]\n{insert_example_text}\n\n---\n\n"

        return output_string


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls) for lbls in labels],
            batch_first=True,
            padding_value=-100,
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def preprocess_dataset(
    dataset: datasets.Dataset,
    num_proc: int,
    block_size: int,
):
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (
            (total_length // block_size) * block_size
            if total_length >= block_size
            else block_size
        )
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    preprocessed_dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return preprocessed_dataset
