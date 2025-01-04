import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import numpy as np
import tqdm
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



try:
    # This library is our indicator that the required installs
    # need to be done.
    import peft
    is_peft_available = True
except ModuleNotFoundError:
    is_peft_available = False

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
residual_stream_component_mapping = {
    "robertaformaskedlm": "roberta.encoder.layer[%s].output"
}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}






def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]
            for i in range(len(batch)):
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                for layer in hidden_layers:
                    # hidden_idx = layer + 1 if layer >= 0 else layer
                    hidden_idx = layer
                    hidden_state = (
                        out.hidden_states[hidden_idx][i][:last_non_padding_index+1]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    hidden_states[layer].append(hidden_state)
            hidden_states
            del out
    return hidden_states
    # return {k: np.vstack(v) for k, v in hidden_states.items()}

def main(
    model_name: str = "/home/ldn/.cache/huggingface/hub/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75",
    seed: int = 42,
    max_length: int = 512,
    dtype: str = "bfloat16"
):
    dtype = dtype_mapping[dtype]
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    
    data_path = "/home/ldn/baidu/pyreft/paddle-version/loreft/datasets/ARC-Challenge/train.json"
    task_dataset = load_dataset(
                    "json", data_files=data_path, split="train"
    )
    
    print(task_dataset)
    trigger_tokens = "the correct answer is "
    
    base_inpus = []
    for i, data in enumerate(task_dataset):
        base_input = "%s\n" % (data["instruction"]) + trigger_tokens + data["answer"] + tokenizer.eos_token
        print(base_input)
    
        base_inpus.append(base_input)
    
    
    

    # load model 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if dtype != "float8" else None,  # save memory
        load_in_8bit=True if dtype == "float8" else False,
        device_map=device
    )
    config = model.config
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
    
    
    
    hidden_layers = [int(layer) for layer in range(33)]
    batch_size = 2
    inputs = base_inpus[:10]
    all_hidden_states = batched_get_hiddens(
        model,
        tokenizer,
        inputs,
        hidden_layers,
        batch_size
    )
    


    sample_nums = 10
    for i in range(sample_nums):
        for key, val in all_hidden_states.items():
            # 获取每一行的最大值
            val[i] = np.max(val[i], axis=1)
            
    # 第0个样本不同层每个token表示最大值
    for i in range(sample_nums):
        representations = []
        for key, val in all_hidden_states.items():
            representations.append(val[i])

        # 通过条件过滤去除大于 10 的值
        representations = np.array(representations)
        representations = representations[0:10,20:40]
        representations_filtered = representations[representations <= 10]
        max_value = np.max(representations_filtered)
        min_value = np.min(representations_filtered)
        fontsize = 12
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), gridspec_kw={'hspace': 0.4})
        print( max_value,min_value)
        s = sns.heatmap(
            representations,
            cmap=mpl.colormaps["Reds"], 
            # center=0.0,
            xticklabels=np.arange(0,representations.shape[1]),
            yticklabels=np.arange(0,representations.shape[0]), 
            square=False,
            cbar_kws={'location': 'right','pad': 0.01},
                vmin=min_value,
                vmax=max_value      # 设置颜色条的最大值)
        )
        ax.invert_yaxis()

        # ax.set_title(f"Layer-Token Representation", fontsize=fontsize, loc="left")
        ax.set_ylabel("Layer", fontsize=fontsize)
        ax.set_xlabel("Token", fontsize=fontsize)
        # ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # print("ytick", s.get_yticklabels())
        s.set_yticklabels(s.get_yticklabels(), rotation=0, horizontalalignment='right')
        fig.savefig(f"./visual_figs/acti_sample{i}.jpg", dpi=200, bbox_inches="tight")
        fig.savefig(f"./visual_figs/acti_sample{i}.pdf", dpi=200, bbox_inches="tight")

        
    
        
        
    
    



if __name__ == "__main__":
    main()