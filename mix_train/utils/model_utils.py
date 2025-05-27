import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_model_info(model_path="", dtype=torch.bfloat16, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    print("=" * 10)
    print(model)


def main():
    print_model_info(model_path="/data/ldn/llm-models/llama-7b")
    print_model_info(model_path="/data/ldn/llm-models/Qwen2.5-7B-Instruct")


if __name__ == "__main__":
    main()
