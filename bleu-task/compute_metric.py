from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import json




def get_rouge_l(answer, generation):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(answer, generation)
    return score['rougeL']





def get_bleu_score(answer, generation):
    generation_tokens = generation.split()
    answer_tokens = answer.split()
    # Compute BLEU score
    bleu_1 = sentence_bleu([answer_tokens], generation_tokens, weights=(1, 0, 0, 0))  # BLEU-1
    # bleu_2 = sentence_bleu([answer_tokens], generation_tokens, weights=(0.5, 0.5, 0, 0))  # BLEU-2
    return bleu_1



def read_json_file(pred_file = "/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/bleu-task/tier_results/-home-ldn-models-gemma-7b.e2e.a6c4ae50-c5cf-11ef-a4c5-7cc2554dc4ec/test_test_outputs.json"):
    # 打开并读取 JSON 文件
    with open(pred_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list
        




def get_avg_score(metric_str="rouge_l"):
    data_list = read_json_file()
    scores = []
    for data in data_list:
        generation = data["generation"].split("The completion is: ")[1].replace("<pad>","")
        answer = data["answer"]
        if metric_str == "rouge_l":
            score = get_rouge_l(answer, generation).precision
        elif metric_str == "bleu":
            score = get_bleu_score(answer, generation)
        scores.append(score)

    scores = [score for score in scores if score > 0.6]
    average = sum(scores) / len(scores)
    average_rounded = round(average, 5)
    print(average_rounded)

if __name__ == '__main__':
    get_avg_score(metric_str="rouge_l")
    get_avg_score(metric_str="bleu")
    