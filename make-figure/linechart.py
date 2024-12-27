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

species = list(without_first_token_data.keys())
accuracy_method_1 = []
accuracy_method_2 = []
for key in species:
    accuracy_method_1.append(without_first_token_data[key])
    accuracy_method_2.append(with_first_token_data[key])
species = [s.replace("eval/","").replace("social_i_qa","siqa").replace("ARC-Easy","ARC-c").replace("ARC-Challenge","ARC-e") for s in species]


print(accuracy_method_1)
print(accuracy_method_2)
print(species)



# 模拟数据
# steps = np.arange(1, 8)  # 假设有10个实验步骤
steps = species

# 计算准确率差距
accuracy_diff = np.array(accuracy_method_1) - np.array(accuracy_method_2)

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(steps, accuracy_method_1, marker='o', label='without first token', color='b', linestyle='-', linewidth=2)
plt.plot(steps, accuracy_method_2, marker='s', label='with first token', color='r', linestyle='--', linewidth=2)

# 添加标题和标签
# plt.title('方法 1 和 方法 2 准确率对比', fontsize=14)
plt.xlabel('Datasets', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# 标注准确率差距
for i in range(len(steps)):
    plt.text(steps[i], accuracy_method_1[i] + 0.01, f'{accuracy_diff[i]:.2f}', ha='center', va='bottom', fontsize=10, color='black')

# 添加图例
plt.legend()

# 显示图形
# plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("visual_figs/linechart.png")
plt.savefig("visual_figs/linechart.pdf")