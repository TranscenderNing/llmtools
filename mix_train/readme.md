# 环境
  Name: peft
  Version: 0.14.0


# 小批量样本测试

```
CUDA_VISIBLE_DEVICES=6 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/Qwen2.5-7B-Instruct \
  --data_dir /data/ldn/datasets \
  --greedy_decoding \
  --max_n_train_example 100 \
  --max_n_eval_example 20 \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13" \
  -m 10 \
  -e 1 \
  > logs/tier-Qwen2.5-7B-Instruct-mix.log 2>&1 &

```

# QWEN模型
```
CUDA_VISIBLE_DEVICES=6 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/Qwen2.5-7B-Instruct \
  --data_dir /data/ldn/datasets \
  --greedy_decoding \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13" \
  -m 10 \
  -e 1 \
  --model_name qwen \
  > logs/tier-Qwen2.5-7B-Instruct-mix.log 2>&1 &
```




# GEMMA模型
```
CUDA_VISIBLE_DEVICES=7 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/gemma-7b \
  --data_dir /data/ldn/datasets \
  --greedy_decoding \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15" \
  -m 10 \
  -e 1 \
  --model_name gemma \
  > logs/tier-gemma-7b-Instruct-mix.log 2>&1 &
```

## math dataset
```
CUDA_VISIBLE_DEVICES=7 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/gemma-7b \
  --data_dir /data/ldn/datasets \
  --task math \
  --greedy_decoding \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15" \
  -m 10 \
  -e 6 \
  --model_name gemma \
  > logs/tier-gemma-7b-Instruct-mix-math-2.log 2>&1 &
```


# LAMMA模型
```
CUDA_VISIBLE_DEVICES=7 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/llama-7b \
  --data_dir /data/ldn/datasets \
  --greedy_decoding \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15" \
  -m 10 \
  -e 1 \
  --model_name llama \
  > logs/tier-llama-7b-Instruct-mix.log 2>&1 &
```


## math dataset
```
CUDA_VISIBLE_DEVICES=5 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/llama-7b \
  --data_dir /data/ldn/datasets \
  --task math \
  --greedy_decoding \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15" \
  -m 10 \
  -e 6 \
  --model_name lamma \
  > logs/tier-lamma-7b-Instruct-mix-math-2.log 2>&1 &
```


# MISTRAL模型
```
CUDA_VISIBLE_DEVICES=6 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/Mistral-7B-v0.1 \
  --data_dir /data/ldn/datasets \
  --greedy_decoding \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15" \
  -m 10 \
  -e 1 \
  --model_name mistral \
  > logs/tier-Mistral-7B-v0.1-Instruct-mix.log 2>&1 &
```



## math dataset
```
CUDA_VISIBLE_DEVICES=6 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/Mistral-7B-v0.1 \
  --data_dir /data/ldn/datasets \
  --task math \
  --greedy_decoding \
  --intervention_type GateLowRankEditor_1 \
  -l "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15" \
  -m 10 \
  -e 6 \
  --model_name mistral \
  > logs/tier-mistral-7b-Instruct-mix-math-2.log 2>&1 &
```
