import matplotlib.pyplot as plt

# 数据
components = ['query_output', 'key_output', 'value_output', 'attention_value_output', 'attention_output', 'mlp_output', 'block_output']
accuracy = [35.58, 32.08, 55.20, 53.84, 53.92, 55.63, 59.13]

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(components, accuracy, color='r', s=100, edgecolor='black')

# 设置标题和标签
# plt.title('Component vs Accuracy', fontsize=14)
plt.xlabel('Component', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)

# 显示图表
plt.xticks(rotation=45, ha="right")  # x轴标签旋转
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig("visual_figs/radar.png")
plt.savefig("visual_figs/radar.pdf")
