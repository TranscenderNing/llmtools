/home/ldn/baidu/benchmarks-dataset/glue/cola/data_1/dev.json

nohup python tier_train.py --model_path /home/ldn/models/google-t5-t5-base --data_dir /home/ldn/baidu/benchmarks-dataset/glue/cola/data_1 --greedy_decoding  --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/cola-t5.log 2>&1 &