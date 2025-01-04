# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np
import json






with_first_token = "/home/ldn/baidu/reft-pytorch-codes/pyreft/examples/loreft/official_results/llama-7b-hf.commonsense.9e6b83f8-baa0-11ef-8e7b-7cc2554dc4ec/eval_results.json"
without_first_token = "/home/ldn/baidu/reft-pytorch-codes/pyreft/examples/loreft/official_results/llama-7b-hf.commonsense.b272fb60-baa0-11ef-8f38-7cc2554dc4ec/eval_results.json"




# 读取数据的函数
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# 读取数据
with_first_token_data = load_json(with_first_token)
without_first_token_data = load_json(without_first_token)
print(without_first_token_data.keys())
del with_first_token_data["n_params"], with_first_token_data["eval/boolq"]
del without_first_token_data["n_params"], without_first_token_data["eval/boolq"]


print(with_first_token_data)
print(without_first_token_data)

for k,v in with_first_token_data.items():
    with_first_token_data[k] = round(v, 2)
for k,v in without_first_token_data.items():
    without_first_token_data[k] = round(v, 2)

methods = [ "without first token", "with first token"]

# 初始化 species 和 accs
species = list(without_first_token_data.keys())
accs = {methods[0]: [], methods[1]:[]}
for key in species:
    accs[methods[0]].append(without_first_token_data[key])
    accs[methods[1]].append(with_first_token_data[key])
species = [s.replace("eval/","").replace("social_i_qa","siqa").replace("ARC-Easy","ARC-c").replace("ARC-Challenge","ARC-e") for s in species]





    





x = np.arange(len(species))  # the label locations
width = 0.45  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, measurement in accs.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Accuracy")
# ax.set_title("Penguin attributes by species")
ax.set_xticks(x + width -0.1, species)
ax.legend(loc="best",)

ax.set_ylim(0.4, 1.0)

plt.show()
plt.savefig("visual_figs/groupbar_1.png")
plt.savefig("visual_figs/groupbar_1.pdf")



