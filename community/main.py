from new_test import CD
import argparse
from sklearn import metrics
import scipy.io as scio
from sklearn.cluster import KMeans



def measure_value(ground_idx, idx):
    ARI = metrics.adjusted_rand_score(ground_idx, idx)
    NMI = metrics.normalized_mutual_info_score(ground_idx, idx)
    ACC = metrics.precision_score(ground_idx, idx, average='macro')
    print(f'ARI-{ARI}\t NMI-{NMI}\t ACC-{ACC}\t')
    # return ARI, NMI, ACC


if __name__ == '__main__':
    data_clusters = {'snd(o)': 3, 'snd(s)': 2, 'WBN': 10, 'WTN': 10, 'MPD': 6, 'CoRA': 3, 'CiteSeer': 3}
    ground_idx = scio.loadmat('./data/new_snd.mat')['true_idx'][0]



    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='new_snd', help='dataset for testing')
    parser.add_argument('--dimensions', type=int, default=10, help='deduced dimensions')
    parser.add_argument('--alpha', type=float, default=0.2, help='constraint s-hht')
    parser.add_argument('--beta', type=float, default=1, help='constraint h-uwt')
    parser.add_argument('--clusters', type=int, default=3, help='number of communities')
    parser.add_argument('--decay', type=float, default=0.1, help='constrain p')
    parser.add_argument('--order', type=int, default=4, help='high-order vertex proximity')
    args = parser.parse_args()

    obj = CD(name=args.dataset, dimensions=args.dimensions, alpha=args.alpha, beta=args.beta, clusters=args.clusters,
             decay=args.decay, order=args.order)

    obj.run()
    x = obj.U
    # print(x)
    estimator = KMeans(n_clusters=3)  # 构造聚类器
    estimator.fit(x)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和
    print('_________________')
    print(label_pred)
    # print()
    print(ground_idx)

    print(measure_value(ground_idx, label_pred))
