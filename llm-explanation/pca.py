import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载 JSON 文件
try:
    with open('impalaQueries.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("文件未找到，请检查文件路径。")
    exit()
except json.JSONDecodeError:
    print("JSON 解码错误，请检查文件格式。")
    exit()

# 提取 attributes
queries = data.get('queries', [])
attributes_list = []

for query in queries:
    attributes = query.get('attributes', {})
    
    # 只保留数值型字段
    numeric_values = {
        'thread_cpu_time_percentage': attributes.get('thread_cpu_time_percentage', -1),
        'thread_network_receive_wait_time': attributes.get('thread_network_receive_wait_time', -1),
        'thread_cpu_time': attributes.get('thread_cpu_time', -1),
        'bytes_streamed': attributes.get('bytes_streamed', -1),
        'memory_spilled': attributes.get('memory_spilled', -1),
        'thread_network_receive_wait_time_percentage': attributes.get('thread_network_receive_wait_time_percentage', -1),
        'num_backends': attributes.get('num_backends', -1),
        'planning_wait_time': attributes.get('planning_wait_time', -1),
        'client_fetch_wait_time_percentage': attributes.get('client_fetch_wait_time_percentage', -1),
        'estimated_per_node_peak_memory': attributes.get('estimated_per_node_peak_memory', -1),
        'client_fetch_wait_time': attributes.get('client_fetch_wait_time', -1),
        'thread_total_time': attributes.get('thread_total_time', -1),
        'thread_network_send_wait_time_percentage': attributes.get('thread_network_send_wait_time_percentage', -1),
        'thread_network_send_wait_time': attributes.get('thread_network_send_wait_time', -1),
        'thread_storage_wait_time': attributes.get('thread_storage_wait_time', -1),
        'thread_storage_wait_time_percentage': attributes.get('thread_storage_wait_time_percentage', -1),
        'num_fragments': attributes.get('num_fragments', -1),
    }
    attributes_list.append(numeric_values)

# 创建 DataFrame
df = pd.DataFrame(attributes_list)

# 标准化数据
scaler = StandardScaler()
try:
    scaled_data = scaler.fit_transform(df)
except ValueError as e:
    print(f"标准化时出错: {e}")
    exit()

# K-means 聚类
try:
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(scaled_data)
    df['cluster'] = kmeans.labels_
except ValueError as e:
    print(f"K-means 聚类时出错: {e}")
    exit()

# PCA 降维
try:
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)
except ValueError as e:
    print(f"PCA 降维时出错: {e}")
    exit()

# 去掉异常值
threshold = 3  # 设定阈值
mask = np.abs(pca_result) < threshold
pca_result_clean = pca_result[mask.all(axis=1)]
clusters_clean = df['cluster'][mask.all(axis=1)]

# 计算轮廓系数
try:
    silhouette_avg = silhouette_score(pca_result_clean, clusters_clean)
    print(f"轮廓系数: {silhouette_avg}")
except ValueError as e:
    print(f"计算轮廓系数时出错: {e}")
    exit()

# 3D 可视化
# 设置画布大小
fig = plt.figure(figsize=(20, 16))  # 宽 10 英寸，高 8 英寸
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_result_clean[:, 0], pca_result_clean[:, 1], pca_result_clean[:, 2],
                     c=clusters_clean, cmap='viridis')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('PCA Result with K-means Clustering')
plt.colorbar(scatter)
ax.view_init(elev=20, azim=30)  # elev: 俯仰角, azim: 旋转角
plt.show()

# 显示最终保留了多少个值
print(f"最终保留的样本数量: {pca_result_clean.shape[0]}")
