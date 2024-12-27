import json

import sys
import io
import json


def txt2json(txt_file, json_file):
    with open(txt_file, 'r', encoding='utf8') as reader, \
        open(json_file, 'w', encoding='utf8') as writer :
        for line in reader:
            items = line.strip().split('||')
            context = items[0]
            completion = items[1].strip('\n')
            x = {}
            x['context'] = context #+ '||'
            x['completion'] = completion
            writer.write(json.dumps(x)+'\n')


def read_ft_file(ft_file):
    ft_samples = []
    with open(ft_file, 'r') as reader:
        for line in reader:
            items = json.loads(line.strip())
            context = items['context']
            completion = items['completion']
            ft_samples.append([context, completion])

    return ft_samples



# for split in ['train', 'valid', 'test']:
#     txt_file = f"/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/gpt2/data/e2e/{split}.txt"
#     json_file = f"/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/gpt2/data/e2e/{split}.json"
#     txt2json(txt_file, json_file)
#     ft_samples = read_ft_file(json_file)
#     print(ft_samples[:10])
    
    
    

for split in ['train', 'valid', 'test']:
    # txt_file = f"/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/gpt2/data/e2e/{split}.txt"
    json_file = f"/home/ldn/baidu/reft-pytorch-codes/learning/llmtools/tier-complex/gpt2/data/e2e/{split}.json"
    # txt2json(txt_file, json_file)
    ft_samples = read_ft_file(json_file)
    print(ft_samples[0])