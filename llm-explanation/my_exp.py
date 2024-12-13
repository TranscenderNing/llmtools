from llm_exp import LlmExplanation
from utils import load_raw_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_name = "/home/ldn/.cache/huggingface/hub/models--yahma--llama-7b-hf/snapshots/cf33055e5df9cc533abd7ea4707bf727ca2ada75"
data_path = "/home/ldn/baidu/pyreft/paddle-version/loreft/datasets/ARC-Challenge/train.json"
output_dir = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/llm-explanation/visual_figs/arc-c"


raw_dataset = load_raw_data(data_path)
llm_exp = LlmExplanation(model_name)
llm_exp.vis_heatmap(raw_dataset, layer_left=1, layer_right=9, output_dir=f"{output_dir}1-9")
llm_exp.vis_heatmap(raw_dataset, layer_left=9, layer_right=17, output_dir=f"{output_dir}9-17")
llm_exp.vis_heatmap(raw_dataset, layer_left=17, layer_right=25, output_dir=f"{output_dir}17-25")
llm_exp.vis_heatmap(raw_dataset, layer_left=25, layer_right=33, output_dir=f"{output_dir}25-33")
