
import numpy as np
import sys
sys.path.append('..')
from ge.classify import read_node_label, Classifier
from ge import LINE
from sklearn.linear_model import LogisticRegression

from utils import *
import argparse

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings, tr_frac=0.1):
    X, Y = read_node_label('labels.txt')
    # tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora",
                        help='dataset for training')
    parser.add_argument('--tr_frac', type=float, default=0.2, help='tr_frac')

    args = parser.parse_args()
    # load_data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(args.dataset)
    idx = np.argmax(labels, axis=1)
    f = open('labels.txt', 'w')
    for i in range(labels.shape[0]):
        f.write(f'{i} {idx[i]} \n')

    adj = adj.toarray()
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
    nx.write_edgelist(G, "test.edgelist", data=[('weight', int)])
    # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
    #                      create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    G = nx.read_edgelist('test.edgelist',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = LINE(G, embedding_size=128, order='second') # ‘first' 'second' 'all'
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings, args.tr_frac)
    plot_embeddings(embeddings)
