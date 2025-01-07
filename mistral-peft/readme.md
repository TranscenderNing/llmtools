
## train


peft_method


cuda 7: lora

nohup python tier_train.py -e 1 -lr 3e-4 -peft_method lora --model_path /home/ldn/models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-mistral-lora.log 2>&1 &


cuda 0: dora

nohup python tier_train.py -e 1 -lr 3e-4 -peft_method dora --model_path /home/ldn/models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-mistral-dora.log 2>&1 &


cuda 3: pissa

nohup python tier_train.py -e 1 -lr 3e-4 -peft_method pissa --model_path /home/ldn/models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-mistral-pissa.log 2>&1 &

nohup python tier_train.py -e 1 -lr 3e-4 -peft_method pissa --model_path /home/ldn/models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-mistral-pissa.log 2>&1 &




# mistral + lora
nohup python tier_train.py -e 1 -lr 3e-4 -peft_method rslora --model_path /home/ldn/models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-mistral-rslora.log 2>&1 &


# gemma + reslora
nohup python tier_train.py -e 3 -lr 3e-4 -peft_method rslora --model_name gemma --model_path /home/ldn/models/gemma-7b  --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-gemma-rslora.log 2>&1 &



# llama + pissa
cd /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft
nohup python tier_train.py -e 3 -lr 3e-4 -peft_method pissa --model_name llama --model_path /home/ldn/models/llama-7b --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-llama-pissa.log 2>&1 &



# mistral + adalora + 

nohup python tier_train.py -e 1 -lr 3e-4 -peft_method adalora --model_path /home/ldn/models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/comon-mistral-adalora.log 2>&1 &




# gemma + loraplus + 
nohup python tier_train.py --data_dir /data/ldn/datasets --model_name gemma -e 1 -lr 3e-4 -peft_method loraplus --model_path /data/ldn/llm-models/gemma-7b --greedy_decoding --max_n_train_example 10 --max_n_eval_example 2  > /data/ldn/llmtools/mistral-peft/comon-gemma-loraplus.log 2>&1 &



# mistral + loraplus + 
nohup python tier_train.py --data_dir /data/ldn/datasets --model_name mistral -e 1 -lr 3e-4 -peft_method loraplus --model_path /data/ldn/llm-models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /data/ldn/llmtools/mistral-peft/comon-mistral-loraplus.log 2>&1 &


# llama + loraplus + 
nohup python tier_train.py --data_dir /data/ldn/datasets --model_name llama -e 1 -lr 3e-4 -peft_method loraplus --model_path /data/ldn/llm-models/llama-7b --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /data/ldn/llmtools/mistral-peft/comon-llama-loraplus-params.log 2>&1 &







nohup python tier_train.py --data_dir /data/ldn/datasets --model_name mistral -e 1 -lr 3e-4 -peft_method dora --model_path /data/ldn/llm-models/Mistral-7B-v0.1 --greedy_decoding --max_n_train_example 10 --max_n_eval_example 2 > /data/ldn/llmtools/mistral-peft/comon-mistral-params.log 2>&1 &

nohup python tier_train.py --data_dir /data/ldn/datasets --model_name gemma -e 1 -lr 3e-4 -peft_method dora --model_path /data/ldn/llm-models/gemma-7b --greedy_decoding --max_n_train_example 10 --max_n_eval_example 2  > /data/ldn/llmtools/mistral-peft/comon-gemma-params-dora.log 2>&1 &






## predict
nohup python tier_predict.py --greedy_decoding --max_n_eval_example 20 --tier_model_path /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/tier_results/-home-ldn-.cache-huggingface-hub-models--yahma--llama-7b-hf-snapshots-cf33055e5df9cc533abd7ea4707bf727ca2ada75.commonsense.b5fb7bd8-bcec-11ef-a997-7cc2554dc4ec > pred.log 2>&1 &





cuda:4
nohup python tier_train.py --data_dir /data/ldn/datasets --model_name gemma -e 1 -lr 3e-4 -peft_method loraplus --model_path /data/ldn/llm-models/gemma-7b --greedy_decoding  > /data/ldn/llmtools/mistral-peft/comon-gemma-loraplus-e1.log 2>&1 &



cuda:5
nohup python tier_train.py --data_dir /data/ldn/datasets --model_name mistral -e 1 -lr 3e-4 -peft_method loraplus --model_path /data/ldn/llm-models/Mistral-7B-v0.1 --greedy_decoding > /data/ldn/llmtools/mistral-peft/comon-mistral-loraplus-e1.log 2>&1 & 