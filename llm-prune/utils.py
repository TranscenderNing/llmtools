import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



dtype_mapping = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

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



def load_orig_model(model_path, dtype, device, tokenizer_len, need_resize=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    if need_resize:
        model.resize_token_embeddings(tokenizer_len)
    return model






def parse_args():
    parser = argparse.ArgumentParser(description="模型参数配置")
    
    parser.add_argument(
        '--model_path', type=str, default="/home/ldn/.cache/huggingface/hub/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75",
        help="Path to the pre-trained model"
    )
    parser.add_argument(
        '--output_dir', type=str, default="./pruned_model_checkpoints/llama7b-pruned",
        help="Directory to store the model output results"
    )
    parser.add_argument(
        '--dtype', type=str, default='bfloat16',
        help="Specify the model's data type, can be 'float32' or 'float16', default is 'float32'"
    )
    parser.add_argument(
        '--list_pruned_blocks', type=str, default="18;19;20;21;22;23;24;25;26;27;28;29",
        help="List of blocks to prune (e.g., '18;19;20;21;22;23;24;25;26;27;28;29')"
    )
    parser.add_argument(
        '--max_length', type=int, default=512,
        help="Maximum sequence length"
    )

    return parser.parse_args()
