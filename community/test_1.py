# import networkx as nx
# import numpy as np
#
#
# def graph2array(G):
#     return nx.to_numpy_array(G)
#
#
# def array2graph(arr):
#     return nx.from_numpy_array(arr)
#
#
# def cosine_similarity_from_array(arr: np.ndarray):
#     S = np.zeros(arr.shape)
#     for node1 in range(arr.shape[0]):
#         for node2 in range(arr.shape[1]):
#             if node1 == node2:
#                 S[node1][node2] = 1.0
#             else:
#                 S[node1][node2] = np.dot(arr[node1], arr[:, node2]) / (
#                         np.linalg.norm(arr[node1]) * np.linalg.norm(arr[:, node2]))
#     return S
#
#
# def cosine_similarity_from_graph(arr):
#     """Better when sparse"""
#     G = array2graph(arr)
#     degrees = nx.degree(G)
#     sets = {node: set(G.neighbors(node)) for node in nx.nodes(G)}
#     laps = np.array(
#         [[float(len(sets[node_1].intersection(sets[node_2]))) / (float(degrees[node_1] * degrees[node_2]) ** 0.5)
#           if node_1 != node_2 else 1.0 for node_1 in nx.nodes(G)]
#          for node_2 in nx.nodes(G)],
#         dtype=np.float64)
#     return laps
#
#
# def test_performance():
#     G = nx.erdos_renyi_graph(1000, 0.1)
#     a = graph2array(G)
#     cosine_similarity_from_array(a)
#     cosine_similarity_from_graph(a)
#
#
# if __name__ == '__main__':
#     G = nx.erdos_renyi_graph(1000, 0.1)
#     a = nx.to_numpy_array(G, nodelist=list(range(1000)))
#     G = nx.from_numpy_array(a)
#     b = nx.to_numpy_array(G, nodelist=list(range(1000)))
#     print(np.linalg.norm(a-b))
#
#     a = np.array([[0,1,1], [1,0,0], [1,0,0]])
#     G = nx.from_numpy_array(a)
#     b = nx.to_numpy_array(G)
#     print(np.linalg.norm(a-b))
#
#
import numpy as np
from sklearn.cluster import KMeans

data = np.random.rand(100,
                      3)  # 生成一个随机数据，样本大小为100, 特征数为3
# #假如我要构造一个聚类数为3的聚类器
estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(data)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
centroids = estimator.cluster_centers_  # 获取聚类中心
inertia = estimator.inertia_  # 获取聚类准则的总和
print(label_pred)

