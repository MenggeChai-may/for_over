import math
import scipy.io as scio
from copy import deepcopy
import numpy as np
import networkx as nx


class CD(object):
    """
    name: datasets
    dimensions: node deduced dimension
    alpha: s-hh
    beta: h-uw
    times:
    clusters: num of community
    decay:
    order: high-order vertex proximity
    """

    def __init__(self, name, dimensions=64, alpha=0.1, beta=1, times=1, clusters=5, decay=0.1, order=3,
                 max_iteration=300):
        print('Model initialization started.\n')
        self.input = f'./data/{name}.mat'
        self.name = name
        self.lamda = math.pow(10, 9)  # lamda
        self.dimensions = dimensions
        self.alpha = alpha
        self.beta = beta
        self.times = times
        self.clusters = clusters
        self.decay = decay
        self.order = order
        self.max_iteration = max_iteration
        self.converge_threshold = math.pow(10, -3)
        self.lower_control = math.pow(10, -8)  # 控制分母不为零

        self.G = scio.loadmat(self.input)['net']  # 邻接矩阵
        self.number_of_layers = len(self.G)
        self.number_of_nodes = self.G[0].shape[0]

        self.S = self.cosine_similarity()
        self.P = self.high_order_proximity()

        self.current_loss = math.pow(10, 10)
        self.round = 0
        self.V = [self.matrix_random_initialization(self.number_of_nodes, self.dimensions) for i in
                  range(self.number_of_layers)]
        self.U = self.matrix_random_initialization(self.number_of_nodes, self.dimensions)
        self.H = self.matrix_random_initialization(self.number_of_nodes, self.clusters)
        self.W = self.matrix_random_initialization(self.clusters, self.dimensions)

    def cosine_similarity(self):
        S = [np.zeros_like(self.G[0], dtype=np.float64) for i in range(self.number_of_layers)]
        for layer in range(self.number_of_layers):
            for node1 in range(self.number_of_nodes):
                for node2 in range(self.number_of_nodes):
                    if node1 == node2:
                        S[layer][node1][node2] = 1.0
                    else:
                        S[layer][node1][node2] = np.dot(self.G[layer][node1], self.G[layer][:, node2]) / (
                                np.linalg.norm(self.G[layer][node1]) * np.linalg.norm(
                            self.G[layer][:, node2]) + self.lower_control)
        return S

    def cosine_similarity_from_graph(self):
        """
        Better when sparse
        This method uses to deal with adj filled with {0,1}
        """
        laps = []
        for i in range(self.number_of_layers):
            graph = self.array2graph(self.G[i])
            degrees = nx.degree(graph)
            sets = {node: set(graph.neighbors(node)) for node in nx.nodes(graph)}
            lap = np.array(
                [[float(len(sets[node_1].intersection(sets[node_2]))) / (
                        float(degrees[node_1] * degrees[node_2]) ** 0.5)
                  if node_1 != node_2 else 1.0 for node_1 in nx.nodes(graph)]
                 for node_2 in nx.nodes(graph)],
                dtype=np.float64)
            laps.append(lap)
        return laps

    def high_order_proximity(self):
        P = [np.zeros_like(self.G[0], dtype=np.float64) for i in range(self.number_of_layers)]
        for layer in range(self.number_of_layers):
            for i in range(self.order):
                if i == 0:
                    A_t = deepcopy(self.G[layer])
                else:
                    A_t = np.matmul(deepcopy(A_t), self.G[layer])
                P[layer] += np.float64(self.decay ** i) * A_t
        return P

    @staticmethod
    def matrix_random_initialization(n_components, n_features):
        return np.random.uniform(0, 1, [n_components, n_features])

    @staticmethod
    def graph2array(G):
        return nx.to_numpy_array(G)

    @staticmethod
    def array2graph(arr):
        return nx.from_numpy_array(arr)

    def re_matrix_random_initialization(self):
        self.current_loss = math.pow(10, 10)
        self.round = 0
        self.V = [self.matrix_random_initialization(self.number_of_nodes, self.dimensions) for i in
                  range(self.number_of_layers)]
        self.U = self.matrix_random_initialization(self.number_of_nodes, self.dimensions)
        self.H = self.matrix_random_initialization(self.number_of_nodes, self.clusters)
        self.W = self.matrix_random_initialization(self.clusters, self.dimensions)

    def update_rule(self):
        # Update V
        for layer in range(self.number_of_layers):
            numerator_v = np.matmul(self.P[layer], self.U)
            denominator_v = np.matmul(self.V[layer], np.matmul(self.U.T, self.U))
            denominator_v = np.maximum(np.float(self.lower_control), denominator_v)
            self.V[layer] = np.multiply(self.V[layer], numerator_v / denominator_v)

        # Update U
        numerator_u = self.beta * np.matmul(self.H, self.W)
        denominator_u = self.beta * np.matmul(self.U, np.matmul(self.W.T, self.W))
        for layer in range(self.number_of_layers):
            numerator_u += np.matmul(self.P[layer], self.V[layer])
            denominator_u += np.matmul(self.U, np.matmul(self.V[layer].T, self.V[layer]))
        denominator_u = np.maximum(np.float(self.lower_control), denominator_u)
        self.U = np.multiply(self.U, numerator_u / denominator_u)

        # Update W
        numerator_w = np.matmul(self.H.T, self.U)
        denominator_w = np.matmul(self.W, np.matmul(self.U.T, self.U))
        denominator_w = np.maximum(np.float(self.lower_control), denominator_w)
        self.W = np.multiply(self.W, numerator_w / denominator_w)

        # Update H
        numerator_h = (self.alpha * self.number_of_layers + self.lamda) * np.matmul(self.H, np.matmul(self.H.T, self.H))
        denominator_h = 2 * (self.alpha * self.number_of_layers + self.lamda) * np.matmul(self.H,
                                                                                          np.matmul(self.H.T,
                                                                                                    self.H))
        denominator_h = np.maximum(np.float(self.lower_control), denominator_h)
        deta = 0
        for layer in range(self.number_of_layers):
            deta += 4 * self.alpha * np.matmul(self.S[layer], self.H)
        numerator_h = np.multiply(numerator_h, deta + 2 * self.beta * np.matmul(self.U, self.W.T) + (
                4 * self.lamda - 2 * self.beta) * self.H)

        self.H = np.multiply(self.H, np.sqrt(np.sqrt(numerator_h)) / np.sqrt(denominator_h))

    def loss(self):
        item1 = item2 = 0
        for layer in range(self.number_of_layers):
            item1 += np.linalg.norm(self.P[layer] - np.matmul(self.V[layer], self.U.T)) ** 2
            item2 += self.alpha * np.linalg.norm(self.S[layer] - np.matmul(self.H, self.H.T)) ** 2
        item3 = self.beta * np.linalg.norm(self.H - np.matmul(self.U, self.W.T))

        constraint = self.lamda * np.linalg.norm(np.matmul(self.H.T, self.H) - np.eye(self.clusters)) ** 2
        loss = np.float64(item1 + item2 + item3 + constraint)
        print(f'{self.round}\t{item1}\t{item2}\t{item3}\t{constraint}\t{loss}')
        return loss

    def run(self):
        for epoch in range(self.max_iteration):
            self.round = epoch + 1
            self.update_rule()

            loss = self.loss()
            if abs(self.current_loss - loss) < self.converge_threshold:
                break
            if loss < self.converge_threshold:
                self.current_loss = loss
