#!/bin/bash

# 循环从第1层到第28层
for ((i=0; i<28; i++)); do
    # 设置输出目录，包含当前层号
    output_dir="/data/ldn/llm-models/pruned_model_checkpoints/Qwen2.5-7B-Instruct-pruned-block-${i}"
    
    # 执行剪枝命令，仅剪枝当前层
    echo "Pruning block $i, output to ${output_dir}"
    CUDA_VISIBLE_DEVICES=2 python prune_block.py \
        --list_pruned_blocks "$i" \
        --model_path "/data/ldn/llm-models/Qwen2.5-7B-Instruct" \
        --output_dir "$output_dir"
done

echo "All pruning jobs completed."