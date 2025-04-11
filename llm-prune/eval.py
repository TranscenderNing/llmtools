import os
import argparse

os.environ["WANDB_MODE"] = "disabled"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from transformers import Trainer

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, AdaLoraConfig
import re
import argparse


# lora_path = '/home/ldn/baidu/reft-pytorch-codes/gsm8k-test/lora_model' 



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

def split_ds(full_test_ds, seed=42, test_size=0.3):
    # ds = load_dataset("openai/gsm8k", "main")
    # full_test_ds = ds["test"]
    print("full_test_ds", full_test_ds)
    split = full_test_ds.train_test_split(
            test_size=test_size, 
            seed=seed,
            shuffle=True
        )
    print("split ds", split)
    eval_subset = split["test"]
    print("eval_subset", eval_subset)
    return eval_subset

def predict(model_path, seeds = [], batch_size = 16) -> None:
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.padding_side)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    
    seeds_results = []
    for seed in seeds:
        print(f"seed is {seed}")
        ds = load_dataset("openai/gsm8k", "main")
        full_test_ds = ds["test"]
        test_ds = split_ds(full_test_ds, seed=seed)
        
        
        # # for llama
        # def modify_question(example):
        #     messages = [
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": example['question']}
        #     ]
        #     example['question'] = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        #     return example
        # for qwen2.5
        def modify_question(example):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example['question']}
            ]
            messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": example['question']}
            ]
            example['question'] = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
            return example
    
        


        # 使用 map 应用该函数
        test_ds = test_ds.map(modify_question)
        print(test_ds)
        print(test_ds[0])
        
        
        batched_test_ds = test_ds.batch(batch_size)

        # 打印查看批次数据
        correct_count = 0
        print("anwser and generation")
        for batch in batched_test_ds:
            model_inputs = tokenizer(batch["question"], return_tensors="pt", padding="longest").to('cuda')
            generated_ids = model.generate(**model_inputs,max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for answer, raw_generation in zip(batch["answer"], response):
                answer = answer.split("####")[-1].strip()
                print(raw_generation)
                generation = extract_answer_number(sentence=raw_generation)
                print(answer, generation)
                if (
                    abs(float(extract_answer_number(answer)) - generation)
                    <= 0.001
                ):
                    correct_count += 1
            print(f"correct count is : {correct_count}")


        print(f"seed: {seed}, Accuracy: {correct_count / len(test_ds)}")
        seeds_results.append({
            "seed": seed,
            "acc": {correct_count / len(test_ds)}
        })
    
    print(seeds_results)



def get_args():


    # 创建解析器
    parser = argparse.ArgumentParser(description="Add model_path parameter")
    # 添加model_path参数
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model")


    parser.add_argument('--batch_size',
                        type=int,
                        default=32,  # 默认值
                        help="Batch size for processing (default: 32)")
        
    # 解析命令行参数
    args = parser.parse_args()
    return args

def main() -> None:
    # split_ds()
    seeds = [42, 99, 94, 70, 91, 27, 21, 65, 26, 48]
    args = get_args()
    # 获取model_path参数
    model_path = args.model_path
    print(f"Model path is: {model_path}")
    predict(model_path,seeds,args.batch_size)


if __name__ == "__main__":
    main()