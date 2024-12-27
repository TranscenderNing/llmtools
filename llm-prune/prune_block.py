import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
from utils import *
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def copy_weight(model, model_orig, list_pruned_blocks):
    connect_info = {}  # connect_info['TO-small'] = 'FROM-orig'
    connect_info["model.embed_tokens.weight"] = "model.embed_tokens.weight"
    connect_info["model.norm.weight"] = "model.norm.weight"
    connect_info["lm_head.weight"] = "lm_head.weight"
    k = 0
    for k_orig in range(model_orig.config.__getattribute__("num_hidden_layers")):
        if k_orig in list_pruned_blocks:  # uncopied = pruned blocks
            continue
        connect_info[f"model.layers.{k}."] = f"model.layers.{k_orig}."
        print(f"original model.layers.{k_orig} --> pruned model.layers.{k}")
        k = k + 1

    print(f" ** excluded blocks {list_pruned_blocks}")

    t0 = time.perf_counter()
    for k in model.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])
                break
        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
            model.state_dict()[k].copy_(model_orig.state_dict()[k_orig])
    print(f"copy time --- {(time.perf_counter()-t0):.1f} sec")

    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    list_pruned_blocks = args.list_pruned_blocks.split(";")
    list_pruned_blocks = [int(b) for b in list_pruned_blocks]
    print("list_pruned_blocks", list_pruned_blocks)
    num_pruned_blocks = len(list_pruned_blocks)
    
    args.dtype = dtype_mapping[args.dtype]
    
    tokenizer, need_resize = load_tokenizer(args.model_path, max_length=512)
    orig_model = load_orig_model(args.model_path, args.dtype, device, len(tokenizer), need_resize=need_resize)

    prune_model_config = copy.deepcopy(orig_model.config)
    print(f"# blocks before pruning: {prune_model_config.num_hidden_layers}")
    prune_model_config.__setattr__(
        "num_hidden_layers", (prune_model_config.num_hidden_layers - num_pruned_blocks)
    )
    print(f"# blocks after pruning: {prune_model_config.num_hidden_layers}")
    model_pruned = AutoModelForCausalLM.from_config(prune_model_config)
    print(f" pruned model: {model_pruned}")
    prune_model = copy_weight(model_pruned, orig_model, list_pruned_blocks)
    prune_model.save_pretrained(args.output_dir, max_shard_size="10GB")
    tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()