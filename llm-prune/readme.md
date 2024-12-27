python prune_block.py --list_pruned_blocks 18;19;20; --output_dir', type=str, default="./pruned_model_checkpoints/llama7b-pruned",



python prune_block.py --list_pruned_blocks "18;19;20;21;22" --output_dir ./pruned_model_checkpoints/llama7b-pruned-5blocks




python prune_block.py --model_path /home/ldn/models/Mistral-7B-v0.1 --list_pruned_blocks "18;19;20;" --output_dir ./pruned_model_checkpoints/Mistral-pruned-3blocks

