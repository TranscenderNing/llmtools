from utils import parse_json_result

def main():
    prune_7 = "./tier_results/-home-ldn-baidu-reft-pytorch-codes-learning-llmtools-llm-prune-pruned_model_checkpoints-llama7b-pruned-3blocks.commonsense.b1bfa2e4-bf92-11ef-9456-7cc2554dc4ec/eval_results.json"
    latex_str, result_dict = parse_json_result(prune_7)
    print(latex_str)
    
    prune_3 = "./tier_results/-home-ldn-baidu-reft-pytorch-codes-learning-llmtools-llm-prune-pruned_model_checkpoints-llama7b-pruned-7blocks.commonsense.cf3d4fe0-bf85-11ef-9487-7cc2554dc4ec/eval_results.json"
    latex_str, result_dict = parse_json_result(prune_3)
    print(latex_str)


if __name__ == '__main__':
    main()