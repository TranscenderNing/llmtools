from torch.utils.data import Dataset
from collections import defaultdict
from copy import deepcopy
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import os
import torch
import pickle

IGNORE_INDEX = -100


class NormalPeftNLGDataset(Dataset):
    def __init__(
        self,
        task: str,
        data_path: str,
        tokenizer,
        data_split="train",
        dataset=None,
        seed=42,
        max_n_example=None,
        task_config=None,
        method="",
        model_name="",
    ):
        super(NormalPeftNLGDataset, self).__init__()

        self.tokenizer = tokenizer
        self.task = task
        self.data_path = data_path
        self.data_split = data_split
        self.data_path = os.path.join(self.data_path, self.data_split + ".json")
        self.dataset = dataset
        self.seed = seed
        self.max_n_example = max_n_example
        self.task_prompt_template = task_config[self.task]["task_prompt_template"]
        self.trigger_tokens = task_config[self.task]["trigger_tokens"]

        self.task_dataset = self.load_dataset()

        # tokenize and intervene
        # self.result = []
        # for id, data_item in enumerate(tqdm(self.task_dataset)):
        #     tokenized = self.tokenize(data_item, id)
        #     self.result.append(tokenized)
        
        self.result = []
        self.cache_path = os.path.join(data_path, f"model_name.{model_name}.method.{method}.max_n_example.{self.max_n_example}.split.{self.data_split}.cache")
        print(self.cache_path)
        # 如果缓存文件存在，则加载缓存数据
        if os.path.exists(self.cache_path):
            print("Loading from cache...")
            with open(self.cache_path, 'rb') as cache_file:
                self.result = pickle.load(cache_file)
        else:
            print("Processing and caching results...")
            for id, data_item in enumerate(tqdm(self.task_dataset)):
                tokenized = self.tokenize(data_item, id)
                self.result.append(tokenized)

            # 保存结果到缓存文件
            with open(self.cache_path, 'wb') as cache_file:
                pickle.dump(self.result, cache_file)
                


    def load_dataset(self):
        print(self.data_path)
        task_dataset = load_dataset("json", data_files=self.data_path, split="train")
        # select n random examples if specificed
        if self.max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=self.seed)
            task_dataset = task_dataset.select(range(self.max_n_example))
        # testing mode needs raw dataset(not tokenized)
        self.raw_dataset = task_dataset if self.data_split != "train" else None
        return task_dataset

    def tokenize(self, data_item, id):
        result = {}
        # get base prompt(question)  and base input(question + answer)
        if self.task == "commonsense" or self.task == "ARC-Challenge":
            base_prompt = self.task_prompt_template % (data_item["instruction"])
            base_input = (
                base_prompt
                + self.trigger_tokens
                + data_item["answer"]
                + self.tokenizer.eos_token
            )
        elif self.task == "math":
            base_prompt = self.task_prompt_template % (data_item["instruction"])
            base_input = base_prompt + data_item["output"] + self.tokenizer.eos_token
        else:
            raise ValueError(f"Unrecognized task: {self.task}")

        # tokenize base  prompt (question)
        base_prompt_ids = self.tokenizer(
            base_prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)

        # tokenize base input (question + answer) in training mode
        if self.data_split == "train":
            base_input_ids = self.tokenizer(
                base_input,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"][0]
            output_ids = deepcopy(base_input_ids)
            output_ids[:base_prompt_length] = IGNORE_INDEX
            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
        # testing mode and no label
        else:
            result["input_ids"] = base_prompt_ids
            result["id"] = id

        
        return result

    def __len__(self):
        return len(self.result)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return deepcopy(self.result[i])
