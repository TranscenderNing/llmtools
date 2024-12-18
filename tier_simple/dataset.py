import os
from copy import deepcopy
import torch
import transformers
from datasets import load_dataset
from collections import defaultdict
import abc
from tqdm import tqdm
import copy
from typing import Dict
from task_config import task_config
from templates import alpaca_prompt_no_input_template, IGNORE_INDEX
import pickle
from torch.utils.data import Dataset
from transformers import DataCollator
from dataclasses import dataclass
from typing import Sequence

class CommonDataset(Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        task: str,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train",
        dataset=None,
        seed=42,
        max_n_example=None,
        **kwargs,
    ):
        super(CommonDataset, self).__init__()
        result = defaultdict(list)

        # setup
        self.tokenizer = tokenizer
        self.task = task
        self.data_path = data_path
        self.data_split = data_split
        self.dataset = dataset
        self.seed = seed
        self.max_n_example = max_n_example
        self.pad_mode = "first"

        # load the dataset
        self.preprocess(kwargs)
        self.task_dataset = self.load_dataset()

        # kwargs settings
        self.postprocess(kwargs)

        # # tokenize
        # self.result = []
        # for i, data_item in enumerate(tqdm(self.task_dataset)):
        #     tokenized = self.tokenize(data_item)
        #     tokenized["id"] = i
        #     self.result.append(tokenized)
        
        
        # cache tokenize
        cache_file =  self.data_path + '_' + self.data_split + '_tokenized.pkl'
        print('cache file is:', cache_file)
        # 检查缓存文件是否存在
        try:
            with open(cache_file, 'rb') as f:
                # 如果缓存文件存在，加载缓存数据
                self.result = pickle.load(f)
                print("加载缓存数据成功！")
        except FileNotFoundError:
            # 如果缓存文件不存在，重新生成数据并保存
            self.result = []
            for i, data_item in enumerate(tqdm(self.task_dataset)):
                tokenized = self.tokenize(data_item)
                tokenized["id"] = i
                self.result.append(tokenized)
    
            # 保存数据到缓存文件
            with open(cache_file, 'wb') as f:
                pickle.dump(self.result, f)
            print("数据处理完成并缓存！")
            

    @abc.abstractmethod
    def tokenize(self, data_item, **kwargs):
        """How to tokenize a single data item. Override this function!"""
        return

    def preprocess(self, kwargs):
        """Preprocessing."""
        return

    def postprocess(self, kwargs):
        """Postprocessing."""
        return

    def __len__(self):
        return len(self.result)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.result[i])

    def load_dataset(self):
        """Load the dataset (or a portion of it) from HF or a local file."""
        # load the dataset
        if self.dataset is None:
            print("loading data for dataset: ", self.data_path)
            if self.data_path is None:
                task_dataset = load_dataset(self.task, split=self.data_split)
            elif self.data_path.endswith(".json"):
                task_dataset = load_dataset(
                    "json", data_files=self.data_path, split="train"
                )
            else:
                task_dataset = load_dataset(
                    self.task, self.data_path, split=self.data_split
                )
        else:
            task_dataset = self.dataset

        # select n random examples if specificed
        if self.max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=self.seed)
            task_dataset = task_dataset.select(range(self.max_n_example))

        # save raw_dataset pointer for access raw strings
        self.raw_dataset = task_dataset if self.data_split != "train" else None
        return task_dataset




class CommonSupervisedDataset(CommonDataset):

    def preprocess(self, kwargs):
        print(kwargs)
        # basic setup
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        dataset_config = task_config[self.task]
        self.task_prompt_template = dataset_config["task_prompt_template"]
        self.trigger_tokens = dataset_config["trigger_tokens"]
        self.original_data_split = self.data_split
        self.test_split = kwargs["test_split"] if "test_split" in kwargs else None

        # where to pull dataset from
        # instruction-tuning tasks should all eval on alpaca_eval
        if (
            self.task in ["alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair"]
            and self.data_split != "train"
        ):
            self.task = "tatsu-lab/alpaca_eval"
            self.data_path = "alpaca_eval"
            self.data_split = "eval"
        if self.task in ["gsm8k"]:
            self.data_path = "main"  # huggingface dir.
            if self.data_split != "test":
                self.data_split = (
                    "train"  # we split l300 examples from train for validation.
                )
        elif self.task in ["math", "commonsense", "ultrafeedback", "alpaca", "ARC-Challenge"]:
            self.data_path = os.path.join(self.data_path, self.data_split + ".json")

    def postprocess(self, kwargs):
        original_dataset_size = len(self.task_dataset)
        if (
            self.task in ["gsm8k"]
            and self.original_data_split == "train"
            and self.test_split == "validation"
        ):
            self.task_dataset = self.task_dataset.select(
                range(original_dataset_size - 300)
            )
        if self.task in ["gsm8k"] and self.original_data_split == "validation":
            self.task_dataset = self.task_dataset.select(
                range(original_dataset_size - 300, original_dataset_size)
            )
        self.raw_dataset = self.task_dataset  # also update the raw dataset pointer.
        return

    def tokenize(self, data_item):
        result = {}

        # set up prompt
        if self.task == "commonsense" or self.task == "ARC-Challenge":
            base_prompt = self.task_prompt_template % (data_item["instruction"])
            base_input = (
                base_prompt
                + self.trigger_tokens
                + data_item["answer"]
                + self.tokenizer.eos_token
            )
        elif self.task == "math":  # we strip since these are model generated examples.
            base_prompt = self.task_prompt_template % (data_item["instruction"])
            base_input = base_prompt + data_item["output"] + self.tokenizer.eos_token
        elif self.task in [
            "alpaca",
            "instruct",
            "ultrafeedback",
            "ultrafeedback_pair",
            "tatsu-lab/alpaca_eval",
        ]:
            if "input" not in data_item or data_item["input"] == "":
                base_prompt = alpaca_prompt_no_input_template % (
                    data_item["instruction"]
                )
            else:
                base_prompt = self.task_prompt_template % (
                    data_item["instruction"],
                    data_item["input"],
                )
            if self.task == "ultrafeedback_pair" and self.data_split == "train":
                # base input takes rejected output to steer away from.
                base_input = (
                    base_prompt
                    + data_item["rejected_output"]
                    + self.tokenizer.eos_token
                )
            else:
                base_input = (
                    base_prompt + data_item["output"] + self.tokenizer.eos_token
                )
        elif self.task == "gsm8k":
            if (
                "Meta-Llama-3-8B-Instruct" in self.tokenizer.name_or_path
            ):  # pretty bad workaround for llama-3, forgive me
                system_prompt = "You are a helpful assistant."
                # we remove the BOS, otherwise there will be redundant BOS tokens.
                base_prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": data_item["question"]},
                    ],
                    tokenize=False,
                )[len("<|begin_of_text|>") :]
                base_input = (
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": data_item["question"]},
                            {"role": "assistant", "content": data_item["answer"]},
                        ],
                        tokenize=False,
                    )[len("<|begin_of_text|>") :]
                    + self.tokenizer.eos_token
                )
            else:  # setup is from https://github.com/yxli2123/LoftQ/
                base_prompt = f"{data_item['question']}{QUESTION_PROMPT}"
                # note: we remove the extra space here to keep the format clean.
                base_input = (
                    base_prompt
                    + f"{data_item['answer']}{self.tokenizer.eos_token}".replace(
                        "####", "The final answer is: "
                    )
                )
        else:
            raise ValueError(f"Unrecognized task: {self.task}")

        # tokenize
        base_prompt_ids = self.tokenizer(
            base_prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        if self.original_data_split == "train":
            base_input_ids = self.tokenizer(
                base_input,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"][0]

            if self.task == "ultrafeedback_pair" and self.data_split == "train":
                # base output takes chosen output to steer towards to.
                base_output = (
                    base_prompt + data_item["chosen_output"] + self.tokenizer.eos_token
                )

                base_output_ids = self.tokenizer(
                    base_output,
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"][0]
                output_ids = base_output_ids
                output_ids[:base_prompt_length] = IGNORE_INDEX

                # padding! needs to be cautious here. let's unpack:
                # pad inputs with pad_token_id so that attention masks can ignore these tokens.
                # pad outputs with IGNORE_INDEX so that loss calculation can ignore these tokens.
                # and the goal is to have input and output have the same length.
                max_length = max(base_input_ids.size(0), output_ids.size(0))
                input_pad_length = max_length - base_input_ids.size(0)
                output_pad_length = max_length - output_ids.size(0)

                input_pad_tensor = torch.full(
                    (input_pad_length,), self.tokenizer.pad_token_id, dtype=torch.long
                )
                output_pad_tensor = torch.full(
                    (output_pad_length,), IGNORE_INDEX, dtype=torch.long
                )

                base_input_ids = torch.cat((base_input_ids, input_pad_tensor), dim=0)
                output_ids = torch.cat((output_ids, output_pad_tensor), dim=0)
            else:
                output_ids = deepcopy(base_input_ids)
                output_ids[:base_prompt_length] = IGNORE_INDEX

            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
        else:
            # print("Assuming test split for now")
            result["input_ids"] = base_prompt_ids

        return result
    
    


@dataclass
class CommonDataCollator(object):
    """Collate examples for ReFT."""

    data_collator: DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        return batch_inputs
