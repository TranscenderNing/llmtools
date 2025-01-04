# nohup python kmeans.py > kmeans.log 2>&1 &


import numpy as np
import tqdm
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

def get_data_representations(model_name, data_path, max_length=2048, batch_size = 32):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    data = [item['instruction'].split('Answer format:')[0].strip().replace('\n', ' ') for item in data]
    total = len(data)
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
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,  # save memory
        device_map=device,
    )
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    
    batched_inputs = [
        data[p : p + batch_size] for p in range(0, len(data), batch_size)
    ]
    all_representations = []
    num = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            # get the last token, handling right padding if present
            encoded_batch = tokenizer(batch, padding=True, return_tensors="pt")
            encoded_batch = encoded_batch.to(model.device)
            out = model(**encoded_batch, output_hidden_states=True)
            attention_mask = encoded_batch["attention_mask"]
            for i in range(len(batch)):
                num += 1
                print(f'{num} of {total}')
                last_non_padding_index = (
                    attention_mask[i].nonzero(as_tuple=True)[0][-1].item()
                )
                hidden_state = (
                    torch.mean(out.hidden_states[0][i][:last_non_padding_index+1], dim=0)
                    .cpu()
                    .float()
                    .numpy()
                )
                all_representations.append(hidden_state)
            del out
            
        # save all_representations
        print(f'Saving all_representations to representations.npy')
        all_representations_np = np.array(all_representations)
        np.save('representations.npy', all_representations_np)

    return all_representations






def analysize_data_similarity(data_path = 'representations.npy'):
    embeddings =  np.load(data_path)
    print('data shape:', embeddings.shape)
    
    from sklearn.metrics.pairwise import cosine_similarity

    # 计算相似度矩阵（余弦相似度）
    similarity_matrix = cosine_similarity(embeddings)
    
    print('similarity_matrix shape:', similarity_matrix.shape)

    # 选择低相似度的样本：选择与其他样本相似度最低的样本
    selected_indices = []
    remaining_indices = list(range(len(embeddings)))

    # 贪婪采样：每次选择一个与当前选中的样本相似度最小的样本
    while len(selected_indices) < len(embeddings) // 100:  # 假设选取1%的样本
        if len(selected_indices) == 0:
            selected_indices.append(remaining_indices.pop(0))  # 随机选择第一个样本
        else:
            min_similarity = float('inf')
            best_index = None
            for idx in remaining_indices:
                # 计算当前样本与已选择样本的最小相似度
                min_sim = np.min(similarity_matrix[idx, selected_indices])
                if min_sim < min_similarity:
                    min_similarity = min_sim
                    best_index = idx
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

    # 输出选择的高质量且多样化的样本索引
    print(selected_indices)




    
def analysize_data(data_path = 'representations.npy'):

    # 生成随机数据
    data =  np.load(data_path)


def main():
    parser = argparse.ArgumentParser(description="处理命令行参数的示例")
    parser.add_argument('--model_name', type=str, help='模型名称', default='/home/ldn/.cache/huggingface/hub/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75')
    parser.add_argument('--data_path', type=str, help='数据文件的路径', default='/home/ldn/baidu/pyreft/paddle-version/loreft/datasets/commonsense_170k/train.json')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用解析的参数
    # get_data_representations(args.model_name, args.data_path)
    # analysize_data()
    
    analysize_data_similarity()
    


if __name__ == '__main__':
    main()