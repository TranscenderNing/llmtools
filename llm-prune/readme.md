

# eval
```
CUDA_VISIBLE_DEVICES=1 nohup python /data/ldn/llmtools/llm-prune/eval.py \
    --model_path /data/ldn/llm-models/Qwen2.5-7B-Instruct \
    --batch_size 64 \
    > logs/noprune_eval.log 2>&1 &
```
# 一键运行脚本
```
nohup ./prune.sh > logs/prune.log 2>&1 &
```

# 评估 gsm8k , 选取30%，多个seed
```

nohup ./eval.sh > logs/eval.log 2>&1 &
nohup ./eval_cuda2.sh > logs/eval_cuda_2.log 2>&1 &
nohup ./eval_cuda3.sh > logs/eval_cuda_3.log 2>&1 &
nohup ./eval_cuda5.sh > logs/eval_cuda_5.log 2>&1 &
nohup ./eval_cuda6.sh > logs/eval_cuda_6.log 2>&1 &
nohup ./eval_cuda7.sh > logs/eval_cuda_7.log 2>&1 &



```



python prune_block.py --list_pruned_blocks 18;19;20; --output_dir', type=str, default="./pruned_model_checkpoints/llama7b-pruned",



python prune_block.py --list_pruned_blocks "18;19;20;21;22" --output_dir ./pruned_model_checkpoints/llama7b-pruned-5blocks




python prune_block.py --model_path /home/ldn/models/Mistral-7B-v0.1 --list_pruned_blocks "18;19;20;" --output_dir ./pruned_model_checkpoints/Mistral-pruned-3blocks

