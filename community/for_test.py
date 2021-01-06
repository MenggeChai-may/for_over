import scipy.io as scio
import numpy as np

dataset = scio.loadmat('./data/1.mat')
# print(dataset)
a1 = dataset['true_idx1']
data = dataset['Net'][0]
# print(len(data))
true_idx = []
for value in a1:
    true_idx.append(value[0])
print(true_idx)
newdataset = './data/100leaves.mat'
scio.savemat(newdataset, {'net': data, 'true_idx': true_idx})
print('finished')
# print(data[0])
G = data
number_of_layers = len(G)
number_of_nodes = G[0].shape[0]

S = [np.zeros_like(G[0], dtype=np.float64) for i in range(number_of_layers)]
for layer in range(number_of_layers):
    for node1 in range(number_of_nodes):
        for node2 in range(number_of_nodes):
            if node1 == node2:
                S[layer][node1][node2] = 1.0
            else:
                S[layer][node1][node2] = np.dot(G[layer][node1], G[layer][:, node2]) / (
                        np.linalg.norm(G[layer][node1]) * np.linalg.norm(
                    G[layer][:, node2]) + 1e-10)
print(S)

