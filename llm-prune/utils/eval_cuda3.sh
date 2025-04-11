#!/bin/bash

# 设置模型路径的基础部分
base_model_path="/data/ldn/llm-models/pruned_model_checkpoints/Qwen2.5-7B-Instruct-pruned-block"

# 循环遍历每个模型的序号
for i in {14..16}
do
    # 构造每个 model_path
    model_path="${base_model_path}-${i}"
    
    # 打印正在处理的模型路径
    echo "Processing model: ${model_path}"
    
    # 运行 eval.py 脚本并传递 model_path
    CUDA_VISIBLE_DEVICES=3 python /data/ldn/llmtools/llm-prune/eval.py --model_path "${model_path}"
    
    # 这里可以根据需要添加其他命令，比如日志记录或者延时
    # sleep 1  # 可选: 如果需要在每次运行后加延时
done
