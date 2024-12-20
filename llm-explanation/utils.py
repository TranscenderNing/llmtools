import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset   
from pyreft import ReftModel


def topk_intermediate_confidence_heatmap(forward_info, topk=5, layer_nums=32, left=0, right=33, dataset_size=100, output_dir=None):
    # 键是layer的索引，值是统计每个样本top-k的token出现次数
    top_k_kv = {_: {} for _ in range(0, layer_nums + 1)}
    for k, v in forward_info.items():
        for layer_idx, tv_pair_list in enumerate(v["top-value_pair"]):
            for top_k_pair in tv_pair_list:
                if top_k_pair[0] in top_k_kv[layer_idx]:
                    top_k_kv[layer_idx][top_k_pair[0]] += 1
                else:
                    top_k_kv[layer_idx][top_k_pair[0]] = 1
    res_top_k = {}
    for k, v in top_k_kv.items():
        counter = Counter(v)
        top_k_per_layer = counter.most_common(topk)
        res_top_k[k] = top_k_per_layer

    selected_keys = list(range(left, right))
    filtered_res_top_k = {key: res_top_k[key] for key in selected_keys}

    keys = list(filtered_res_top_k.keys())
    words = [item[0] for sublist in filtered_res_top_k.values() for item in sublist]
    counts = [item[1] for sublist in filtered_res_top_k.values() for item in sublist]

    heatmap_data = np.array(counts).reshape(len(keys), topk).T
    word_labels = np.array(words).reshape(len(keys), topk).T

    fig, ax = plt.subplots(figsize=((right-left)*2, (right-left)))
    cax = ax.matshow(heatmap_data, cmap='viridis', vmin=0, vmax=dataset_size)
    divider = make_axes_locatable(ax)
    cax_colorbar = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(cax, cax=cax_colorbar)

    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f'L{key}' for key in keys])
    ax.set_yticks(np.arange(topk))
    ax.set_yticklabels([f"{_i}" for _i in range(topk)])
    for i in range(topk):
        for j in range(len(keys)):
            if word_labels[i, j] == '\n':
                ax.text(j, i, '\\n', ha='center', va='center', color='black', fontsize=16)
            else:
                ax.text(j, i, word_labels[i, j], ha='center', va='center', color='black', fontsize=16)

    ax.set_title(f'Top {topk} from Intermediate Hidden States \n (Layer {left}-{right})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Rank')

    plt.tight_layout()
    plt.savefig(output_dir)
    plt.show()



def load_model_and_tokenizer(model_path, dtype=torch.bfloat16, device_map="", load_reft_model=False, reft_model_path="/home/ldn/baidu/reft-pytorch-codes/pyreft/examples/loreft/official_results/llama-7b-hf.ARC-Challenge.405a98a8-baa0-11ef-aad6-7cc2554dc4ec"):
    model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True, device_map=device_map, load_in_8bit=True, torch_dtype=dtype)
    model.eval()
    if load_reft_model:
        model = ReftModel.load(reft_model_path, model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer



def load_raw_data(data_path="/home/ldn/baidu/pyreft/paddle-version/loreft/datasets/ARC-Challenge/train.json"):
    exp_dataset = load_dataset("json", data_files=data_path, split="train")
    trigger_tokens = "the correct answer is "
    exp_dataset = exp_dataset.select([i for i in  range(100)])
    raw_dataset = []
    for i, data in enumerate(exp_dataset):
        # base_input = "%s\n" % (data["instruction"]) + trigger_tokens + data["answer"]
        base_input = "%s\n" % (data["instruction"]) + trigger_tokens
        raw_dataset.append(base_input)
    return raw_dataset
    