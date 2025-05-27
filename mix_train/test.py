from utils import parse_json_result

def main():
    # prune_7 = "./tier_results/-home-ldn-baidu-reft-pytorch-codes-learning-llmtools-llm-prune-pruned_model_checkpoints-llama7b-pruned-3blocks.commonsense.b1bfa2e4-bf92-11ef-9456-7cc2554dc4ec/eval_results.json"
    # latex_str, result_dict = parse_json_result(prune_7)
    # print(latex_str)
    
    # prune_3 = "./tier_results/-home-ldn-baidu-reft-pytorch-codes-learning-llmtools-llm-prune-pruned_model_checkpoints-llama7b-pruned-7blocks.commonsense.cf3d4fe0-bf85-11ef-9487-7cc2554dc4ec/eval_results.json"
    # latex_str, result_dict = parse_json_result(prune_3)
    # print(latex_str)
    
    # mistral_dora = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/tier_results/-home-ldn-models-Mistral-7B-v0.1.commonsense.ee26ac06-c3fb-11ef-8126-7cc2554dc4ec/eval_results.json"
    # latex_str, result_dict = parse_json_result(mistral_dora)
    # print(latex_str)
    
    # mistral_lora = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/tier_results/-home-ldn-models-Mistral-7B-v0.1.commonsense.f872b250-c3fa-11ef-b48e-7cc2554dc4ec/eval_results.json"
    # latex_str, result_dict = parse_json_result(mistral_lora)
    # print(latex_str)
    
    # mistral_pissa = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/tier_results//-home-ldn-models-Mistral-7B-v0.1.commonsense.1ff1fadc-c3fd-11ef-8fd8-7cc2554dc4ec/eval_results.json"
    # latex_str, result_dict = parse_json_result(mistral_pissa)
    # print(latex_str)
    
    
    gemma_dora = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/tier_results/-home-ldn-models-gemma-7b.commonsense.e452e25a-c589-11ef-9bac-7cc2554dc4ec/eval_results.json"
    gemma_lora = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/tier_results/-home-ldn-models-gemma-7b.commonsense.751a373a-c589-11ef-8eda-7cc2554dc4ec/eval_results.json"
    gemma_pissa = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/mistral-peft/tier_results/-home-ldn-models-gemma-7b.commonsense.f70d2f18-c589-11ef-836c-7cc2554dc4ec/eval_results.json"
    
    
    gemma_res = [gemma_dora,gemma_lora,gemma_pissa]
    methods = ["dora", "lora", "pissa"]
    for method, res in zip(methods, gemma_res):
        print("="*100)
        latex_str, result_dict = parse_json_result(res)
        print(method, latex_str)
        
    



if __name__ == '__main__':
    main()