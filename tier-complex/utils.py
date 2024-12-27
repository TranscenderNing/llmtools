import json
import argparse
import importlib
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from datasets import Dataset
from tier_interventions import (DualSpacesIntervention,
                                FeedForwardIntervention, LoRAIntervention,
                                LoreftIntervention, LoreftIntervention_v2,
                                MloraIntervention, MoEIntervention,
                                MosLoRAIntervention,
                                MultiHeadAttentionIntervention,
                                RedIntervention)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq)

# constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

gsm8k_template = """%s\nAnswer the above question. First think step by step and then answer the final number.\n"""

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

alpaca_prompt_template = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

alpaca_prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""


dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
    "torch.bfloat16": torch.bfloat16,
}


intervention_mapping = {
    "LoreftIntervention": LoreftIntervention,
    "LoRAIntervention": LoRAIntervention,
    "MoEIntervention": MoEIntervention,
    "FeedForwardIntervention": FeedForwardIntervention,
    "MultiHeadAttentionIntervention": MultiHeadAttentionIntervention,
    "MosLoRAIntervention": MosLoRAIntervention,
    "RedIntervention": RedIntervention,
    "DualSpacesIntervention": DualSpacesIntervention,
    "LoreftIntervention_v2": LoreftIntervention_v2,
    "MloraIntervention": MloraIntervention,
}


task_config = {
    "ARC-Challenge": {
        "train_datasets": ["ARC-Challenge"],
        "eval_datasets": ["ARC-Challenge"],
        "task_prompt_template": "%s\n",
        "trigger_tokens": "the correct answer is ",
        "generation_args": {
            # align with https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 32,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            },
        },
    },
    "commonsense": {
        "train_datasets": ["commonsense_170k"],
        "eval_datasets": [
            "boolq",
            "piqa",
            "social_i_qa",
            "hellaswag",
            "winogrande",
            "ARC-Easy",
            "ARC-Challenge",
            "openbookqa",
        ],
        "task_prompt_template": "%s\n",
        "trigger_tokens": "the correct answer is ",
        "generation_args": {
            # align with https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 32,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            },
        },
    },
    "math": {
        "train_datasets": ["math_10k"],
        # "eval_datasets": [
        #     "MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq",
        # ],
        "eval_datasets": ["AQuA", "gsm8k", "mawps", "SVAMP"],
        "task_prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # slightly changed to optimize our performance on top of
            # https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 512,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            },
        },
    },
    "alpaca": {
        "train_datasets": ["alpaca_data_cleaned"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "instruct": {
        "train_datasets": ["instruct"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "ultrafeedback": {
        "train_datasets": ["ultrafeedback"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "ultrafeedback_pair": {
        "train_datasets": ["argilla/ultrafeedback-binarized-preferences-cleaned"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "glue": {
        "train_datasets": None,
        "eval_datasets": None,
        "task_prompt_template": None,
        "trigger_tokens": None,
    },
    "gsm8k": {
        "train_datasets": ["gsm8k"],
        "eval_datasets": ["gsm8k"],
        "task_prompt_template": gsm8k_template,
        "trigger_tokens": "First think step by step and then answer the final number.\n",
        "generation_args": {
            # default values are from LoftQ
            # https://arxiv.org/pdf/2310.08659.pdf
            True: {
                "max_new_tokens": 256,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "do_sample": True,
            },
        },
    },
}


def get_args():
    parser = argparse.ArgumentParser(
        description="A simple script that takes different arguments."
    )

    parser.add_argument("-task", "--task", type=str, default="commonsense")
    parser.add_argument(
        "-data_dir",
        "--data_dir",
        type=str,
        default="/home/ldn/baidu/pyreft/paddle-version/loreft/datasets",
    )
    parser.add_argument("-train_dataset", "--train_dataset", type=str, default=None)
    parser.add_argument("-eval_dataset", "--eval_dataset", type=str, default=None)
    parser.add_argument(
        "-model_path",
        "--model_path",
        type=str,
        help="yahma/llama-7b-hf",
        default="/home/ldn/.cache/huggingface/hub/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75",
    )
    parser.add_argument("-seed", "--seed", type=int, help="42", default=42)
    parser.add_argument("-l", "--layers", type=str, help="2;10;18;26", default="all")
    parser.add_argument("-r", "--rank", type=int, help=8, default=8)
    parser.add_argument("-p", "--position", type=str, help="f1+l1", default="f7+l7")
    parser.add_argument("-e", "--epochs", type=int, help="1", default=6)
    parser.add_argument("-is_wandb", "--is_wandb", action="store_true")
    parser.add_argument("-wandb_name", "--wandb_name", type=str, default="reft")
    parser.add_argument("-save_model", "--save_model", action="store_true")
    parser.add_argument(
        "-max_n_train_example", "--max_n_train_example", type=int, default=None
    )
    parser.add_argument(
        "-max_n_eval_example", "--max_n_eval_example", type=int, default=None
    )
    parser.add_argument(
        "-type",
        "--intervention_type",
        type=str,
        help="LoreftIntervention",
        default="LoreftIntervention",
    )
    parser.add_argument(
        "-gradient_accumulation_steps",
        "--gradient_accumulation_steps",
        type=int,
        default=2,
    )
    parser.add_argument("-batch_size", "--batch_size", type=int, default=8)
    parser.add_argument("-eval_batch_size", "--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "-output_dir", "--output_dir", type=str, default="./tier_results"
    )
    parser.add_argument("-lr", "--lr", type=float, default=9e-4)
    parser.add_argument("-schedule", "--schedule", type=str, default="linear")
    parser.add_argument("-wu", "--warmup_ratio", type=float, default=0.1)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.00)
    parser.add_argument("-dropout", "--dropout", type=float, default=0.00)
    parser.add_argument("-act_fn", "--act_fn", type=str, default=None)
    parser.add_argument("-add_bias", "--add_bias", action="store_true")
    parser.add_argument("-test_split", "--test_split", type=str, default="test")
    parser.add_argument("-train_on_inputs", "--train_on_inputs", action="store_true")
    parser.add_argument("-max_length", "--max_length", type=int, help=512, default=512)
    parser.add_argument("-nt", "--use_normalized_template", action="store_true")
    parser.add_argument("-allow_cls_grad", "--allow_cls_grad", action="store_true")
    parser.add_argument(
        "-metric_for_best_model",
        "--metric_for_best_model",
        type=str,
        default="accuracy",
    )
    parser.add_argument(
        "-dtype",
        "--dtype",
        type=str,
        default="bfloat16",
    )
    parser.add_argument(
        "-logging_steps", "--logging_steps", type=int, help=1, default=1
    )
    parser.add_argument("-wandb_dir", "--wandb_dir", type=str, default="wandb")
    parser.add_argument("-wandb_proj", "--wandb_proj", type=str, default="MyReFT")
    parser.add_argument("-sw", "--share_weights", action="store_true")
    parser.add_argument("-gd", "--greedy_decoding", action="store_true")

    # decoding params
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument("-top_p", "--top_p", type=float, default=None)
    parser.add_argument("-top_k", "--top_k", type=float, default=None)

    # lora add-ons
    parser.add_argument("-disable_reft", "--disable_reft", action="store_true")
    parser.add_argument("-use_lora", "--use_lora", action="store_true")
    parser.add_argument("-lora_rank", "--lora_rank", type=int, default=8)
    parser.add_argument("-lora_alpha", "--lora_alpha", type=int, default=32)
    parser.add_argument("-lora_modules", "--lora_modules", type=str, default="o_proj")
    parser.add_argument(
        "-lora_layers",
        "--lora_layers",
        type=str,
        help="2;10;18;26",
        default="2;10;18;26",
    )

    args = parser.parse_args()
    return args


def get_pred_args():
    parser = argparse.ArgumentParser(
        description="A simple script that takes different arguments."
    )
    parser.add_argument("-task", "--task", type=str, default="commonsense")
    parser.add_argument(
        "-data_dir",
        "--data_dir",
        type=str,
        default="/home/ldn/baidu/pyreft/paddle-version/loreft/datasets",
    )
    parser.add_argument("-eval_dataset", "--eval_dataset", type=str, default=None)
    parser.add_argument(
        "-model_path",
        "--model_path",
        type=str,
        help="yahma/llama-7b-hf",
        default="/home/ldn/.cache/huggingface/hub/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75",
    )
    parser.add_argument(
        "-tier_model_path",
        "--tier_model_path",
        type=str,
        help="tier_model_path",
        default="/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/tier_results/-home-ldn-.cache-huggingface-hub-models--yahma--llama-7b-hf-snapshots-cf33055e5df9cc533abd7ea4707bf727ca2ada75.commonsense.5c3d8600-bc78-11ef-b30b-7cc2554dc4ec",
    )
    parser.add_argument("-seed", "--seed", type=int, help="42", default=42)
    parser.add_argument(
        "-max_n_eval_example", "--max_n_eval_example", type=int, default=None
    )
    parser.add_argument("-eval_batch_size", "--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "-output_dir", "--output_dir", type=str, default="./tier_results"
    )
    parser.add_argument("-test_split", "--test_split", type=str, default="test")
    parser.add_argument("-max_length", "--max_length", type=int, help=512, default=512)
    parser.add_argument(
        "-dtype",
        "--dtype",
        type=str,
        default="bfloat16",
    )
    # decoding params
    parser.add_argument("-gd", "--greedy_decoding", action="store_true")
    parser.add_argument("-t", "--temperature", type=float, default=None)
    parser.add_argument("-top_p", "--top_p", type=float, default=None)
    parser.add_argument("-top_k", "--top_k", type=float, default=None)
    args = parser.parse_args()
    return args


def load_tokenizer(model_path, max_length=512):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False
    return tokenizer, need_resize


def load_base_model(model_path, dtype, device, tokenizer_len, need_resize=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    print("model", model)
    if need_resize:
        model.resize_token_embeddings(tokenizer_len)
    return model, model.config


def getattr_for_torch_module(model, parameter_name):
    """Recursively fetch the model based on the name."""
    current_module = model
    for param in parameter_name.split("."):
        if "[" in param:
            current_module = getattr(current_module, param.split("[")[0])[
                int(param.split("[")[-1].strip("]"))
            ]
        else:
            current_module = getattr(current_module, param)
    return current_module


def get_module_hook(model, representation) -> nn.Module:
    """Render the intervening module with a hook."""
    hook_type = "register_forward_hook"
    parameter_name = f'model.layers[{representation["layer"]}]'
    module = getattr_for_torch_module(model, parameter_name)
    module_hook = getattr(module, hook_type)
    return module_hook


class HandlerList:
    """General class to set hooks and set off hooks."""

    def __init__(self, handlers):
        self.handlers = handlers

    def __len__(self):
        return len(self.handlers)

    def remove(self):
        for handler in self.handlers:
            handler.remove()

    def extend(self, new_handlers):
        self.handlers.extend(new_handlers.handlers)
        return self


#  gather hidden states on intervention locations
def gather_neurons(tensor_input, unit_locations_as_list):
    unit_locations = torch.tensor(unit_locations_as_list, device=tensor_input.device)
    tensor_output = torch.gather(
        tensor_input,
        1,
        unit_locations.reshape(
            *unit_locations.shape, *(1,) * (len(tensor_input.shape) - 2)
        ).expand(-1, -1, *tensor_input.shape[2:]),
    )
    return tensor_output


# Replace selected neurons in `tensor_input` by `replacing_tensor_input`.
def scatter_neurons(
    tensor_input,
    replacing_tensor_input,
    unit_locations_as_list,
):

    unit_locations = torch.tensor(unit_locations_as_list, device=tensor_input.device)

    # [1,1,4096]
    meta_component = (
        torch.arange(tensor_input.shape[-1]).unsqueeze(dim=0).unsqueeze(dim=0)
    )
    start_index, end_index = (
        meta_component.min().tolist(),
        meta_component.max().tolist() + 1,
    )

    # 4096
    # last_dim = meta_component.shape[-1]
    # 0, 1, 2, ..., batch_size-1
    _batch_idx = torch.arange(tensor_input.shape[0]).unsqueeze(1)
    tensor_input[
        _batch_idx, unit_locations, start_index:end_index
    ] = replacing_tensor_input
    return tensor_input


# do intervention
def do_intervention(
    base_representation,
    intervention,
):
    """Do the actual intervention."""
    # base_representation： 从隐藏状态抽取出的对应token的隐藏状态 f7+l7: batch_size, 14, hidden_size
    # intervention: 干预网络
    intervened_representation = intervention(
        base_representation,
    )
    return intervened_representation


# Introducing corresponding classes based on strings
def get_type_from_string(type_str):
    """Help function to convert string to type"""
    # Remove <class ' and '> from the string
    type_str = type_str.replace("<class '", "").replace("'>", "")

    # Split the string into module and class name
    module_name, class_name = type_str.rsplit(".", 1)

    # Import the module
    module = importlib.import_module(module_name)

    # Get the class
    cls = getattr(module, class_name)

    return cls


def create_directory(path):
    """Create directory if not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directory '{path}' created successfully.")
    else:
        logging.info(f"Directory '{path}' already exists.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.seed(seed)


def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@dataclass
class TierDataCollator(object):
    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][
            ..., :max_seq_length
        ]
        return batch_inputs


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance.

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        return ""


def extract_output(pred, trigger=""):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ""
    output = pred[start + len(trigger) :].lstrip()  # left strip any whitespaces
    return output


def make_dataloader(
    dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn
    )


def compute_metrics(
    task: str,
    dataset_name: str,
    intervenable,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    data_items: list,
    trigger_tokens: str,
    batch_size: int = 4,
    data_collator=None,
    greedy_decoding=False,
    temperature=None,
    top_p=None,
    top_k=None,
    device=None,
):
    # switch the tokenizer mode first for generation tasks
    if task != "glue":
        tokenizer.padding_side = "left"  # switch padding side for collator
        num_beams = (
            4
            if task in ["commonsense", "math", "ARC-Challenge"] and not greedy_decoding
            else 1
        )

    eval_dataloader = make_dataloader(
        eval_dataset, batch_size, data_collator, shuffle=False
    )
    correct_count = 0
    total_count = 0
    generations = []
    eval_iterator = tqdm(eval_dataloader, position=0, leave=True)

    if (
        "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path
    ):  # pretty bad workaround for llama-3, forgive me
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        trigger_tokens = "assistant\n\n"

    with torch.no_grad():
        for _, inputs in enumerate(eval_iterator):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # [layers, batch_size, positions]
            intervention_locations = inputs["intervention_locations"].permute(1, 0, 2)
            # get left padding count, [batch_size], and add to locations
            left_padding = (inputs["input_ids"] == tokenizer.bos_token_id).nonzero(
                as_tuple=True
            )[1]
            if left_padding.numel() > 0:
                left_padding = left_padding.reshape(1, -1, 1).to(
                    device
                )  # [1, batch_size, 1]
                intervention_locations += left_padding
                intervention_locations -= 1  # offset for the sink padding
            else:
                print("Warning: No BOS token found, skipping left padding adjustment.")

                # repeat each batch by num_beams times in intervention locations
                # -> [layers, batch_size * num_beams, positions]
            intervention_locations = intervention_locations.repeat_interleave(
                num_beams, dim=1
            ).tolist()

            # set generation args depending on task
            generation_args = {
                "base": {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                "unit_locations": intervention_locations,
                "intervene_on_prompt": True,
                "eos_token_id": tokenizer.eos_token_id,
                "early_stopping": True,
            }
            if "generation_args" in task_config[task]:
                generation_args.update(
                    task_config[task]["generation_args"][greedy_decoding]
                )
            if (
                "Meta-Llama-3-8B-Instruct" in tokenizer.name_or_path
            ):  # pretty bad workaround for llama-3, forgive me
                generation_args["eos_token_id"] = terminators

            # override generation args if necessary
            if temperature is not None:
                generation_args["temperature"] = temperature
            if top_p is not None:
                generation_args["top_p"] = top_p
            if top_k is not None:
                generation_args["top_k"] = top_k

            # generate with intervention on prompt
            _, steered_response = intervenable.generate(**generation_args)

            # detokenize in batch
            actual_preds = tokenizer.batch_decode(
                steered_response, skip_special_tokens=True
            )

            for id, pred in zip(inputs["id"].tolist(), actual_preds):
                example = data_items[id]
                try:
                    raw_generation = extract_output(pred, trigger_tokens)
                except:
                    print("get not split based on trigger tokens: ", raw_generation)
                    raw_generation = "WRONG"

                # check if generation is correct
                if task == "commonsense" or task == "ARC-Challenge":
                    answer = example["answer"]
                    generation = raw_generation[:]
                    if generation.strip() == answer.strip():
                        correct_count += 1
                elif task == "math":
                    answer = example["answer"]
                    answer = answer.strip()
                    if not is_float(answer):  # assuming this is from AQuA:
                        generation = extract_answer_letter(raw_generation)
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    else:
                        generation = extract_answer_number(raw_generation)
                        if abs(float(answer) - generation) <= 0.001:
                            correct_count += 1

                # log
                total_count += 1
                metric_str = round(correct_count / total_count, 3)
                eval_iterator.set_postfix({"em": metric_str})
                instruction = (
                    example["question"] if task == "gsm8k" else example["instruction"]
                )
                generations += [
                    {
                        "instruction": instruction,
                        "raw_generation": raw_generation,
                        "generation": generation,
                        "answer": answer,
                    }
                ]
    return generations, {f"eval/{dataset_name}": correct_count / total_count}



def parse_json_result(file_path = "./tier_results/-home-ldn-baidu-reft-pytorch-codes-learning-llmtools-llm-prune-pruned_model_checkpoints-llama7b-pruned-3blocks.commonsense.b1bfa2e4-bf92-11ef-9456-7cc2554dc4ec/eval_results.json"):
    with open(file_path, "r") as f:
        acc_dict = json.load(f)
    latex_str = []
    nums = list(acc_dict.values())
    nums.append(sum(nums) / len(nums)) 
    nums = [num * 100 for num in nums]
    print(nums)
    nums_str = [f"{num:.1f}" for num in nums]
    latex_str = ' & '.join(nums_str)
    return latex_str, acc_dict
    
    

