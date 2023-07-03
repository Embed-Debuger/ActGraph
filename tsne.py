import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

matplotlib.use('Agg')
import re
# from matplotlib import font_manager

def tsne_reduce_dimention(X):
    Y = PCA(n_components=10).fit_transform(X)
    Y = tsne(Y, 2, 10, 50, 1000)
    return Y

def Create_Save_Tsne(X, labels, name, Path=None):
    # tsne降维可视化，并保存图片
    # 参数 置信度 标签 标签名 图片保存路径
    # Y = tsne(X, 2, 50, 20.0)
    # Y = tsne(X, 2, 10, 50, 500)

    # X_pca = PCA(n_components=len(X[0])).fit_transform(X)
    # Y = TSNE(n_components=2, verbose=1).fit_transform(X_pca)

    # Y = PCA(n_components=10).fit_transform(X)
    Y = tsne(X, 2, 10, 50, 1000)

    fig, ax = plt.subplots(figsize=(10, 10))
    # plt.rcParams['savefig.dpi'] = 500  # 图片像素
    # plt.rcParams['figure.dpi'] = 500  # 分辨率
    markers = ["o", "v", "s", "p", "*"]
    colors = ['b', 'g', 'r', 'c', 'm']
    for i in range(len(list(set(labels)))):
        label = i
        idx = np.where(labels==label)
        y1 = Y[idx, 0]
        y2 = Y[idx, 1]
        color = colors[label]   # 颜色
        marker = markers[label] # 标记
        ax.scatter(y1, y2, s=100, c=color, marker=marker)
    # ax.legend(name, loc="best", ncol=1, fontsize=20)
    # # 加x y
    # plt.xlabel('X', fontsize=18)
    # plt.ylabel('Y', fontsize=16)

    # a, b = scatter.legend_elements()
    # # 对b进行更改。填入标签的英文名
    #
    # i = 0
    # print(b)
    #
    # while i < len(b):
    #     print(len(b))
    #     print(i)
    #     str1 = re.sub("\D", "", b[i])  # 提取这一行中的数字
    #     # print(name[int(str1)])
    #     b[i] = "".join(b[i].replace(str1, name[int(str1)].replace("\n", "")).split())  # 去除换行，去除空格
    #     i = i + 1
    #
    # legend1 = plt.legend(a, b, fontsize=8, loc="upper right")  # prop 使用特定字体
    # ax.add_artist(legend1)

    if Path != None:
        fig.savefig(Path)
    # plt.show()
    # plt.close()


def tsne(x, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """
    t-SNE
    @x: N*D的矩阵
    @no_dims： 降维后的维数
    @initial_dims： 使用PCA预降维的维数
    @perplexity： 困惑度
    @max_iter：迭代次数
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # 初始化参数
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # 对称化
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # early exaggeration
    P = P * 4
    P = np.maximum(P, 1e-12)

    for iter in range(max_iter):
        # 计算低维概率分布
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # 计算梯度
        PQ = P - Q
        for i in range(n):
            dy[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # 更新梯度
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # 计算Cost
        if (iter + 1) % 100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum(P / 4 * np.log(P / 4 / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y


def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''
        二分搜索寻找beta，并计算出成对的prob
    '''
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    pair_prob = np.zeros((n, n))  # n*n的概率矩阵
    beta = np.ones((n, 1))  # n*1的参数矩阵
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." % (i, n))

        betamin = -np.inf
        # betamax = np.max
        betamax = np.inf  # 添加
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])
        # print(betamax)
        # 二分搜索，寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1

        # 记录prob值
        pair_prob[i,] = this_prob

    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return pair_prob


def cal_perplexity(dist, idx=0, beta=1.0):
    '''
    计算perplexity
    @dist：计算出的距离矩阵，
    @idx：指dist中自己的位置
    @beta：是高斯分布参数
    '''
    prob = np.exp(- dist * beta)
    prob[idx] = 0
    sum_prob = np.sum(prob)
    perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
    prob /= sum_prob

    return perp, prob


def pca(X, no_dims=50):
    '''
        使用PCA预降维到no_dims维
    '''
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    print("M shape:", M.shape)
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def cal_pairwise_dist(x):
    '''
        计算X_i,X_j的距离
        (Xi - Xj)^2 = Xi^2 + Xj^2 - 2XiXj
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)

    return dist


if __name__ == "__main__":
    Create_Save_Tsne(X=np.array([[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                                 [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]),
                     labels=[0, 1, 2], name=["a", "b", "c", "d", "e", "f"], Path="11")

    # from sklearn.manifold import TSNE
    # from sklearn.decomposition import PCA
    # data = np.array([[0, 0, 1],
    #                  [1, 0, 0],
    #                  [0, 1, 0]])
    # # data_pca = PCA(n_components=len(data[0])).fit_transform(data)
    # data_tsne = TSNE(n_components=2, verbose=1).fit_transform(data)


    print("end")
