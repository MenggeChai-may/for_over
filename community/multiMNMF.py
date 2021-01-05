import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

import argparse
import scipy.io as scio


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# NMI, ARI, PUR计算
def testValue(ground_idx, idx):
    ARI = metrics.adjusted_rand_score(ground_idx, idx)
    NMI = metrics.normalized_mutual_info_score(ground_idx, idx)
    PUR = purity_score(ground_idx, idx)
    return (NMI, ARI, PUR)


def mmnmf(data, l, k, alpha, tol=1e-2, maxiter=300, pdist=1e12):
    """
    l 节点降维程度
    k 社团数目
    """
    to_layer = len(data)  # 网络层数
    nodenum = data[0].shape[0]  # 节点数
    m = [np.mat(np.random.rand(nodenum, l)) for i in range(to_layer)]
    u = [np.mat(np.random.rand(nodenum, l)) for i in range(to_layer)]
    c = [np.mat(np.random.rand(k, l)) for i in range(to_layer)]
    h = np.mat(np.random.rand(nodenum, k))

    for it in range(maxiter):

        temp = 0
        for layer in range(to_layer):
            m[layer] = np.multiply(m[layer], np.divide(data[layer] * u[layer], m[layer] * u[layer].T * u[layer] + 1e-7))
            u[layer] = np.multiply(u[layer], np.divide(data[layer] * m[layer] + h * c[layer],
                                                       u[layer] * m[layer].T * m[layer] + alpha * u[layer] * c[
                                                           layer].T * c[layer] + 1e-7))
            c[layer] = np.multiply(c[layer], np.divide(h.T * u[layer], c[layer] * u[layer].T * u[layer] + 1e-7))
            temp += u[layer] * c[layer].T

        h = np.multiply(h, np.divide(temp + h * h.T * h, h + h * h.T * temp + 1e-7))

        if it % 10 == 0:
            dist = 0
            for j in range(to_layer):
                dist += np.linalg.norm(data[j] - m[j] * u[j].T) ** 2 + alpha * np.linalg.norm(h - u[j] * c[j].T) ** 2
            if pdist - dist < tol:
                break
            pdist = dist
            print(pdist)

        h = np.array(h)
        norms = h.sum(axis=1)
        norms[norms == 0] = 1
        h = h / norms[:, np.newaxis]

        return h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.2, help='parameter')
    parser.add_argument('--numk', type=int, default=3, help='num of communities')
    parser.add_argument('--dimension', type=int, default=20, help='demension of node embedding')
    parser.add_argument('--dataset', type=str, default='snd(o)_solved', help='dataset for testing')

    args = parser.parse_args()


    net = scio.loadmat(args.dataset)['net'][0]
    label = scio.loadmat(args.dataset)['idx']
    # label有时需要处理
    tru_idx = []
    for value in label:
        tru_idx.append(value[0])
    print(tru_idx)
    h = mmnmf(net, args.dimension, args.numk, args.alpha)
    # idx = np.argmax(h, axis=1)
    estimator = KMeans(args.numk)
    estimator.fit(h)
    idx = estimator.labels_
    print(idx)
    print(testValue(tru_idx, idx))


