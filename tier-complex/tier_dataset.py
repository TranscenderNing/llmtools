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


class TierNLGDataset(Dataset):
    def __init__(
        self,
        task: str,
        data_path: str,
        tokenizer,
        data_split="train",
        dataset=None,
        seed=42,
        max_n_example=None,
        position="f7+l7",
        num_interventions=None,
        task_config=None,
        model_name=""
    ):
        super(TierNLGDataset, self).__init__()

        self.tokenizer = tokenizer
        self.first_n, self.last_n = self.parse_positions(position)
        self.task = task
        self.data_path = data_path
        self.data_split = data_split
        self.data_path = os.path.join(self.data_path, self.data_split + ".json")
        self.dataset = dataset
        self.seed = seed
        self.max_n_example = max_n_example
        self.num_interventions = num_interventions
        self.task_prompt_template = task_config[self.task]["task_prompt_template"]
        self.trigger_tokens = task_config[self.task]["trigger_tokens"]

        self.task_dataset = self.load_dataset()

        # tokenize and intervene
        # self.result = []
        # for id, data_item in enumerate(tqdm(self.task_dataset)):
        #     tokenized = self.tokenize(data_item, id)
        #     self.result.append(tokenized)
        
        self.result = []
        self.cache_path = os.path.join(data_path, f"model_name.{model_name}.position.{position}.interventions.{self.num_interventions}.max_n_example.{self.max_n_example}.split.{self.data_split}.cache")
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
                

    def parse_positions(self, positions: str):
        # parse position
        first_n, last_n = 0, 0
        if "+" in positions:
            first_n = int(positions.split("+")[0].strip("f"))
            last_n = int(positions.split("+")[1].strip("l"))
        else:
            if "f" in positions:
                first_n = int(positions.strip("f"))
            elif "l" in positions:
                last_n = int(positions.strip("l"))
        return first_n, last_n

    def load_dataset(self):
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
        last_position = base_prompt_length

        # compute intervention locations field
        intervention_locations = self.get_intervention_locations(
            first_n=self.first_n,
            last_n=self.last_n,
            last_position=last_position,
            num_interventions=self.num_interventions,
        )
        result["intervention_locations"] = intervention_locations
        result["id"] = id

        # add a single padding token before input_ids and fix everything
        result["input_ids"] = torch.cat(
            (
                torch.tensor(
                    [
                        self.tokenizer.pad_token_id,
                    ]
                ),
                result["input_ids"],
            )
        )
        result["attention_mask"] = (
            result["input_ids"] != self.tokenizer.pad_token_id
        ).int()
        if "labels" in result:
            result["labels"] = torch.cat(
                (
                    torch.tensor(
                        [
                            IGNORE_INDEX,
                        ]
                    ),
                    result["labels"],
                )
            )
        # intervention_locations
        result["intervention_locations"] = (
            torch.IntTensor(result["intervention_locations"]) + 1
        ).tolist()

        return result

    def __len__(self):
        return len(self.result)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return deepcopy(self.result[i])

    # layers * intervention tokens
    def get_intervention_locations(
        self, first_n, last_n, last_position, num_interventions
    ):
        _first_n, _last_n = first_n, last_n
        assert (
            first_n + last_n <= last_position
        ), "first_n + last_n should be smaller than last_position"
        first_n = min(last_position // 2, _first_n)
        last_n = min(last_position // 2, _last_n)

        pad_amount = (_first_n - first_n) + (_last_n - last_n)
        pad_position = -1

        position_list = (
            [i for i in range(first_n)]
            + [i for i in range(last_position - last_n, last_position)]
            + [pad_position for _ in range(pad_amount)]
        )
        intervention_locations = [position_list] * num_interventions

        return intervention_locations
