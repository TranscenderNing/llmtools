for qwen2.5 model

# /data/ldn/llm-models/Qwen2.5-7B-Instruct




## few samples train
```
CUDA_VISIBLE_DEVICES=1 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/Qwen2.5-7B-Instruct \
  --data_dir /data/ldn/datasets \
  --greedy_decoding \
  --max_n_train_example 100 \
  --max_n_eval_example 20 \
  --intervention_type GateLowRankEditor \
  --m 10 \
  -e 3 \
  > logs/tier-Qwen2.5-7B-Instruct-gate-edit.log 2>&1 &
```


```

CUDA_VISIBLE_DEVICES=1 nohup python tier_train.py \
  --model_path /data/ldn/llm-models/Qwen2.5-7B-Instruct \
  --data_dir /data/ldn/datasets \
  --greedy_decoding \
  --intervention_type GateLowRankEditor \
  --rank 8 \
  --m 10 \
  -e 3 \
  > logs/tier-Qwen2.5-7B-Instruct-gate-edit.log 2>&1 &

```




## train
python tier_train.py


### commonsense 
nohup python tier_train.py --model_path /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/llm-prune/pruned_model_checkpoints/llama7b-pruned --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/comon-prune-lora.log 2>&1 &












nohup python tier_train.py --model_name mistral --model_path /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/llm-prune/pruned_model_checkpoints/Mistral-pruned-3blocks --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20 -e 3  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/comon-prune-lore31block-mistral.log 2>&1 &

RedIntervention +++++++

nohup python tier_train.py --model_path /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/llm-prune/pruned_model_checkpoints/llama7b-pruned --greedy_decoding --max_n_train_example 100 -type RedIntervention --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/comon-prune-red.log 2>&1 &

nohup python tier_train.py -task commonsense \
-data_dir /home/ldn/baidu/pyreft/paddle-version/loreft/datasets \
-model yahma/llama-7b-hf \
-seed 42 \
-l all -r 8 -p l7 -e 6 -lr 9e-4 \
-type LoreftIntervention \
-gradient_accumulation_steps 2 \
-batch_size 8 \
-eval_batch_size 8 \
--dropout 0.00 \
--test_split test \
--use_normalized_template \
--share_weights \
--greedy_decoding > /home/ldn/baidu/reft-pytorch-codes/logs/comon-l7.log 2>&1 &


## predict
nohup python tier_predict.py --greedy_decoding --max_n_eval_example 20 --tier_model_path /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/tier_results/-home-ldn-.cache-huggingface-hub-models--yahma--llama-7b-hf-snapshots-cf33055e5df9cc533abd7ea4707bf727ca2ada75.commonsense.b5fb7bd8-bcec-11ef-a997-7cc2554dc4ec > pred.log 2>&1 &









