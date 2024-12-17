python myexp.py

tier_train.py


# commonsense 
nohup python tier_train.py --greedy_decoding --max_n_train_example 100 --max_n_eval_example 20  > /home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/comon-.log 2>&1 &




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


