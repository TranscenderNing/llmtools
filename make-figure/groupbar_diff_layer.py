import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ['PIQA', 'SIQA', 'ARC-c']  # 数据集
first_16_layers = [82.1, 78.5, 62.0]  # first 16 layers 对应的数据
last_16_layers = [74.0, 59.3, 36.5]  # last 16 layers 对应的数据

# 设置条形图的位置和宽度
bar_height = 0.25  # 设置条形的高度
index = np.arange(len(datasets))  # 定义条形图的位置

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))  # 可以调整图表大小

# 绘制每个任务的条形图
bar1 = ax.barh(index - bar_height, first_16_layers, bar_height, label='First 16 Layers',)
bar2 = ax.barh(index, last_16_layers, bar_height, label='Last 16 Layers', )

# 添加标签和标题
ax.set_xlabel('Accuracy (%)')
ax.set_xlim(30, 85)
ax.set_ylabel('Datasets')
# ax.set_title('Accuracy for Different Layers and Datasets')
ax.set_yticks(index)  # 设置y轴的刻度
ax.set_yticklabels(datasets)  # 设置y轴的标签
ax.legend()

# 调整子图布局，避免重叠
plt.tight_layout()
plt.show()
plt.savefig("visual_figs/groupbar_2.png")
plt.savefig("visual_figs/groupbar_2.pdf")