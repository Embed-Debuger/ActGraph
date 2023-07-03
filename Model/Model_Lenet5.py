import keras.backend
from keras import optimizers, backend, losses
from keras.backend import function, gradients
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, BatchNormalization, Conv1D, MaxPooling1D
from keras.models import Model
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from common import *


def lenet5_cifar10(lr=0.01, drop_rate=0, path=None):
    # 构建模型
    x = Input(shape=(32, 32, 3))
    y = Conv2D(filters=6, kernel_size=[5, 5], strides=[1, 1], activation='relu')(x)
    y = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(y)
    y = Conv2D(filters=16, kernel_size=[5, 5], strides=[1, 1], activation='relu')(y)
    y = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(y)
    y = Conv2D(filters=120, kernel_size=[5, 5], strides=[1, 1], activation='relu')(y)
    y = Dropout(drop_rate)(y)
    y = Flatten()(y)
    y = Dense(84, activation='relu')(y)
    y = Dense(10, activation='softmax')(y)
    model = Model(x, y)

    # 模型优化器
    # optimizer = optimizers.Adam(lr=lr)
    optimizer = optimizers.SGD(lr=lr)
    # optimizer = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    try:
        model.load_weights(path)
        print("模型权重加载成功")
    except:
        print("模型权重加载失败")
    return model


def lenet5_mnist(lr=0.01, drop_rate=0, path=None):
    # 构建模型
    x = Input(shape=(28, 28, 1))
    y = Conv2D(filters=6, kernel_size=[3, 3], strides=[1, 1], activation='relu')(x)
    y = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(y)
    y = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], activation='relu')(y)
    y = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(y)
    y = Conv2D(filters=120, kernel_size=[3, 3], strides=[1, 1], activation='relu')(y)
    y = Dropout(drop_rate)(y)
    y = Flatten()(y)
    y = Dense(84, activation='relu')(y)
    y = Dense(10, activation='softmax')(y)
    model = Model(x, y)

    # 模型优化器
    optimizer = optimizers.Adam(lr=lr)
    # optimizer = optimizers.SGD(lr=lr)
    # optimizer = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    try:
        model.load_weights(path)
        print("模型权重加载成功")
    except:
        print("模型权重加载失败")
    return model


class lenet5_cifar10_graph(object):
    def __init__(self, model):
        # 读取模型权重和权重名
        self.model = model
        self.model_weights = self.model.get_weights()
        self.model_weights_names = [weight.name for layer in self.model.layers for weight in layer.weights]

        # 构图初始化
        self.weights_index = [0, 2, 4, 6, 8]  # weights中重要的层的索引
        self.filters_shape = [3, 6, 16, 120, 84, 10]  # 重要层的filter数目
        self.edge_weights = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                             for i in range(1, len(self.filters_shape))]    # 权边初始化
        self.weights_out = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                            for i in range(1, len(self.filters_shape))]     # 将输出和权边相乘（和模型权重及其输出有关）
        self.layers_out = [np.zeros(shape=(1, self.filters_shape[i]))
                           for i in range(1, len(self.filters_shape))]      # 每层输出初始化
        self.edge_num = sum([np.size(ary) for ary in self.edge_weights])    # 权边数
        self.node_num = sum(self.filters_shape)         # 结点数
        self.edge_weights = self.Cal_weights_edges()    # 计算有权连边(仅和模型权重相关)
        self.graph = None   # 图
        self.node_feature = None
        self.A = None   # 邻接矩阵

        # 初始化模型重要层输出
        self.get_conv_1st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[2].output])
        self.get_conv_2st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[4].output])
        self.get_conv_3st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[6].output])
        self.get_dens_1st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[8].output])
        self.get_dens_2st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[9].output])

    def Cal_weights_edges(self):
        # 计算filter之间的连边值
        for idx, edge_weight in enumerate(self.edge_weights):  # weights的第几层
            inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx == 0 or idx == 1 or idx == 2:  # 卷积层的filter连边权重
                for i in range(inshape):
                    for j in range(oushape):
                        self.edge_weights[idx][i][j] = np.linalg.norm(
                            self.model_weights[self.weights_index[idx]][..., i, j])
                        # self.edge_weights[idx][i][j] = 1
            elif idx == 3:  # 卷积和全连接的连边权重
                for i in range(inshape):
                    for j in range(oushape):
                        # 从卷积层到全连接层的转换，多维数组被整为一维数组，默认数据按卷积核数一一对应;（16*5*5-->1*400）
                        # 5*5是对应卷积核大小
                        self.edge_weights[idx][i][j] = np.linalg.norm(
                            self.model_weights[self.weights_index[idx]][..., i * (5 * 5):(i + 1) * (5 * 5), j])
                        # self.edge_weights[idx][i][j] = np.linalg.norm(
                        #     self.model_weights[self.weights_index[idx]][..., i, j])
                        # self.edge_weights[idx][i][j] = 1
            elif idx == 4:  # 全连接和全连接的连边权重
                for i in range(inshape):
                    for j in range(oushape):
                        self.edge_weights[idx][i][j] = self.model_weights[self.weights_index[idx]][i, j]
                        # self.edge_weights[idx][i][j] = 1
            # 对每层权重做 归一化
            self.edge_weights[idx] = normalization(normalization(self.edge_weights[idx]))
            # 对每层权重做 标准化
            # self.edge_weights[idx] = self.standardization(self.edge_weights[idx])
        return self.edge_weights

    def Cal_layers_out(self, sample):
        # 计算每层的filter输出
        # 将filter的输出映射到权重
        conv_1st_output = self.get_conv_1st_output([sample.tolist()])[0]
        conv_2st_output = self.get_conv_2st_output([sample.tolist()])[0]
        conv_3st_output = self.get_conv_3st_output([sample.tolist()])[0]
        dens_1st_output = self.get_dens_1st_output([sample.tolist()])[0]
        dens_2st_output = self.get_dens_2st_output([sample.tolist()])[0]

        # 对模型每层输出取范数
        for idx, edge_weight in enumerate(self.edge_weights):
            # inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx == 0:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(conv_1st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1     # 模型输出归一化
                # self.layers_out[idx] = normalization(self.layers_out[idx]) * 0.1
            elif idx == 1:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(conv_2st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1   # 归一化后设置权重
                # self.layers_out[idx] = normalization(self.layers_out[idx]) * 0.2
            elif idx == 2:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(conv_3st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
                # self.layers_out[idx] = normalization(self.layers_out[idx]) * 0.3
            elif idx == 3:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(dens_1st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
                # self.layers_out[idx] = normalization(self.layers_out[idx]) * 0.6
            elif idx == 4:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(dens_2st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
                # self.layers_out[idx] = normalization(self.layers_out[idx]) * 1

        self.Cal_out_mul_weights()  # 计算模型输出映射到权重的值

    def Cal_out_mul_weights(self):
        # 计算模型输出映射到权重的值
        for i in range(5):
            self.weights_out[i] = np.multiply(self.edge_weights[i], self.layers_out[i])

    def Generate_graph(self, sample, node_fea="EC", threshold=0):
        self.Cal_layers_out(sample)
        # 将模型权边生成图
        # print("generate_graph")
        G = nx.Graph()  # 创建空的简单有向图
        for idx, weight_out in enumerate(self.weights_out):  # weights的第几层
            inshape = weight_out.shape[0]
            oushape = weight_out.shape[1]
            # 计算每层的偏移索引
            if idx == 0:
                offset_in = 0
                offset_ou = offset_in + inshape
                offset_in_old = offset_in
                offset_ou_old = offset_ou
            else:
                offset_in = offset_ou_old
                offset_ou = offset_ou_old + inshape
                offset_in_old = offset_in
                offset_ou_old = offset_ou

            # 构建 节点对 及其 权边
            for i in range(inshape):
                for j in range(oushape):
                    G.add_edge(i + offset_in, j + offset_ou, weight=self.weights_out[idx][i][j])
        G = self.delet_G_edge(G, threshold=threshold)  # 删减连边
        # G = self.delet_G_node(G)     # 删减多余节点
        # G = self.delet_G_other_node(G, min=self.filters_shape[0], max=np.sum(self.filters_shape[:-1])) # 删除无输入或者无输出节点
        self.graph = G
        self.node_feature = self.Cal_feature(node_fea=node_fea)    # 计算图节点特征
        self.node_feature = np.array(self.node_feature).reshape(1, -1)
        self.A = np.array(nx.adjacency_matrix(self.graph).todense())    # 邻接矩阵
        feature = np.dot(self.node_feature, self.A)

        return G, self.A, feature.reshape(-1)

    def Generate_act(self, sample):
        self.Cal_layers_out(sample)
        layer_out = np.concatenate((self.layers_out[-2][0], self.layers_out[-1][0]), axis=0)
        return layer_out

    def Cal_feature(self, node_fea="EC"):
        num = self.node_num
        # 计算图指标
        feature = [0]*num
        if node_fea == "EC":
            iter = 1000
            while True: # 迭代次数不够多会引起报错
                try:
                    # 特征向量中心度
                    EC = nx.eigenvector_centrality(self.graph, max_iter=iter, weight="weight")  # 字典格式
                    break
                except:
                    iter += 1000
                    if iter > 10000:    # 防止一直算不出卡在这
                        EC = np.zeros(num)
                        break
            for i in range(num):
                try:
                    feature[i] = EC[i]
                except:     # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "CC":
            CC = nx.closeness_centrality(self.graph)  # 紧密中心度
            for i in range(num):
                try:
                    feature[i] = CC[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "D":
            D = nx.degree(self.graph)  # 度
            for i in range(num):
                try:
                    feature[i] = D[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "DC":
            DC = nx.degree_centrality(self.graph)  # 度中心度
            for i in range(num):
                try:
                    feature[i] = DC[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "BC":
            BC = nx.betweenness_centrality(self.graph)  # 间接中心性
            for i in range(num):
                try:
                    feature[i] = BC[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue

        return feature

    def Draw_graph(self, save_path=None, show=True):
        plt.rcParams['figure.figsize'] = (10, 10)
        pos = {}
        filter = np.array(self.filters_shape)
        max_filter = np.max(filter)  # 某一层最大的神经元数
        n_color = [(217/255,209/255,155/255), (153/255,164/255,188/255), (101/255,102/255,103/255),
                   (150/255,80/255,75/255), (164/255,152/255,133/255), (215/255,202/255,177/255)]    # 每层神经元的颜色
        node_color = []
        # 设置每个神经元的位置
        for layer in range(filter.size):  # 层号
            base_idx = np.sum(filter[:layer])  # 每层号的base
            # base_y = int(max_filter/2)-int(filter[layer]/2)
            base_y = 0
            jiange = (max_filter / (filter[layer] - 1 + 1e-5))  # 同层节点之间的距离
            for idx in range(filter[layer]):  # 每层的号
                x = layer
                # y = base_y + idx
                y = base_y + int(idx * jiange)
                pos[base_idx + idx] = (x, y)
                node_color.append(n_color[layer])

        node_size = [v * 30 for v in dict(self.graph.degree).values()]
        edge_color=[float(d['weight']) for (u, v, d) in self.graph.edges(data=True)]

        nodes = nx.draw_networkx_nodes(self.graph, pos,
                                       node_size=node_size)
        edges = nx.draw_networkx_edges(self.graph, pos, edge_color=edge_color, width=2,
                                       edge_cmap=plt.cm.cividis)
        # nx.draw(self.graph, pos=pos, with_labels=False,
        #         node_size=node_size,
        #         edge_color=edge_color, width=2)
        plt.colorbar(edges)
        if save_path!=None:
            plt.savefig(save_path)
        if show:
            plt.show()

    def delet_G_edge(self, G, threshold=0):
        # 删除权重小于等于threshold的连边
        for i in list(G.edges()):
            # print(i)
            weight = G[i[0]][i[1]]["weight"]
            if weight <= threshold:  # 如果权边权重小于阈值 ，删除
                G.remove_edge(i[0], i[1])
        return G

    def delet_G_node(self, G):
        # 删除没有连边的节点
        for i in list(G.adj.keys()):
            # print(i)
            if G.adj[i] == {}:  # 节点无连边
                G.remove_node(i)
        return G

    def delet_G_other_node(self, G, min, max):
        # 删除无输入节点
        for node_idx in list(G.adj.keys()):
            if node_idx >= min:
                if (np.array(list(G.adj[node_idx].keys())) > node_idx).all():  # 该图节点不存在输入
                    for next_node_idx in list(G.adj[node_idx].keys()):
                        G.remove_edge(node_idx, next_node_idx)
                    # G.remove_node(node_idx)
        # 删除无输出节点
        for node_idx in list(G.adj.keys()):
            if node_idx < max:
                if (np.array(list(G.adj[node_idx].keys())) < node_idx).all():  # 该图节点不存在输出
                    for next_node_idx in list(G.adj[node_idx].keys()):
                        G.remove_edge(node_idx, next_node_idx)
                    # G.remove_node(node_idx)
        return G

    def get_layer_A(self, graph_layer_num=4):
        # 按层来获取邻接矩阵
        if type(self.A) == type(None):
            print("未用样本激活神经元")
            return self.A
        in_beg = np.sum(self.filters_shape[:graph_layer_num])
        in_end = np.sum(self.filters_shape[:graph_layer_num+1])
        out_beg = in_end
        out_end = np.sum(self.filters_shape[:graph_layer_num+2])
        A = self.A[in_beg:in_end][..., out_beg:out_end]     # 某一层的邻接矩阵
        weight_num = self.weights_index[graph_layer_num]  # 权重层的索引

        return weight_num, A

    def get_layer_node(self, graph_layer_num=4):
        # 获得
        beg = np.sum(self.filters_shape[:graph_layer_num])
        end = np.sum(self.filters_shape[:graph_layer_num+1])
        return self.node_feature[..., beg:end]


class lenet5_mnist_graph(object):
    def __init__(self, model):
        # 读取模型权重和权重名
        self.model = model
        self.model_weights = self.model.get_weights()
        self.model_weights_names = [weight.name for layer in self.model.layers for weight in layer.weights]

        # 构图初始化
        self.weights_index = [0, 2, 4, 6, 8]  # weights中重要的层的索引
        self.filters_shape = [1, 6, 16, 120, 84, 10]  # 重要层的filter数目
        self.edge_weights = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                             for i in range(1, len(self.filters_shape))]    # 权边初始化
        self.weights_out = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                            for i in range(1, len(self.filters_shape))]     # 将输出和权边相乘（和模型权重及其输出有关）
        self.layers_out = [np.zeros(shape=(1, self.filters_shape[i]))
                           for i in range(1, len(self.filters_shape))]      # 每层输出初始化
        self.edge_num = sum([np.size(ary) for ary in self.edge_weights])    # 权边数
        self.node_num = sum(self.filters_shape)         # 结点数
        self.edge_weights = self.Cal_weights_edges()    # 计算有权连边(仅和模型权重相关)
        self.graph = None   # 图
        self.node_feature = None
        self.A = np.zeros(shape=(self.node_num, self.node_num))  # 邻接矩阵上三角
        self.A_T = np.zeros(shape=(self.node_num, self.node_num))  # 邻接矩阵上三角

        # 初始化模型重要层输出
        self.get_conv_1st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[2].output])
        self.get_conv_2st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[4].output])
        self.get_conv_3st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[6].output])
        self.get_dens_1st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[8].output])
        self.get_dens_2st_output = function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[9].output])

    def Cal_weights_edges(self):
        # 计算filter之间的连边值
        for idx, edge_weight in enumerate(self.edge_weights):  # weights的第几层
            inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx == 0 or idx == 1 or idx == 2:  # 卷积层的filter连边权重
                for i in range(inshape):
                    for j in range(oushape):
                        self.edge_weights[idx][i][j] = np.linalg.norm(
                            self.model_weights[self.weights_index[idx]][..., i, j])
                        # self.edge_weights[idx][i][j] = 1
            elif idx == 3:  # 卷积和全连接的连边权重
                for i in range(inshape):
                    for j in range(oushape):
                        # 从卷积层到全连接层的转换，多维数组被整为一维数组，默认数据按卷积核数一一对应;（16*5*5-->1*400）
                        # 5*5是对应卷积核大小
                        # self.edge_weights[idx][i][j] = np.linalg.norm(
                        #     self.model_weights[self.weights_index[idx]][..., i * (5 * 5):(i + 1) * (5 * 5), j])
                        self.edge_weights[idx][i][j] = np.linalg.norm(
                            self.model_weights[self.weights_index[idx]][..., i, j])
                        # self.edge_weights[idx][i][j] = 1
            elif idx == 4:  # 全连接和全连接的连边权重
                for i in range(inshape):
                    for j in range(oushape):
                        self.edge_weights[idx][i][j] = self.model_weights[self.weights_index[idx]][i, j]
                        # self.edge_weights[idx][i][j] = 1
            # 对每层权重做 归一化
            self.edge_weights[idx] = normalization(normalization(self.edge_weights[idx]))
            # 对每层权重做 标准化
            # self.edge_weights[idx] = self.standardization(self.edge_weights[idx])
            # self.edge_weights[idx] = np.where(self.edge_weights[idx]>=0.1, self.edge_weights[idx], 0)
        return self.edge_weights

    def Cal_layers_out(self, sample):
        # 计算每层的filter输出
        # 将filter的输出映射到权重
        conv_1st_output = self.get_conv_1st_output([sample.tolist()])[0]
        conv_2st_output = self.get_conv_2st_output([sample.tolist()])[0]
        conv_3st_output = self.get_conv_3st_output([sample.tolist()])[0]
        dens_1st_output = self.get_dens_1st_output([sample.tolist()])[0]
        dens_2st_output = self.get_dens_2st_output([sample.tolist()])[0]

        # 对模型每层输出取范数
        for idx, edge_weight in enumerate(self.edge_weights):
            # inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx == 0:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(conv_1st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1     # 模型输出归一化
            elif idx == 1:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(conv_2st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1   # 归一化后设置权重
            elif idx == 2:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(conv_3st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
            elif idx == 3:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(dens_1st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
            elif idx == 4:
                for i in range(oushape):
                    self.layers_out[idx][0][i] = np.linalg.norm(dens_2st_output[..., i])
                self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
        self.Cal_out_mul_weights()  # 计算模型输出映射到权重的值

    def Cal_out_mul_weights(self):
        # 计算模型输出映射到权重的值
        for i in range(5):
            self.weights_out[i] = np.multiply(self.edge_weights[i], self.layers_out[i])

    def Generate_graph(self, sample, node_fea="D", threshold=0):
        self.Cal_layers_out(sample)
        # 将模型权边生成图
        # print("generate_graph")
        G = nx.Graph()  # 创建空的简单有向图
        for idx, weight_out in enumerate(self.weights_out):  # weights的第几层
            inshape = weight_out.shape[0]
            oushape = weight_out.shape[1]
            # 计算每层的偏移索引
            if idx == 0:
                offset_in = 0
                offset_ou = offset_in + inshape
                offset_in_old = offset_in
                offset_ou_old = offset_ou
            else:
                offset_in = offset_ou_old
                offset_ou = offset_ou_old + inshape
                offset_in_old = offset_in
                offset_ou_old = offset_ou

            # 构建 节点对 及其 权边
            for i in range(inshape):
                for j in range(oushape):
                    G.add_edge(i + offset_in, j + offset_ou, weight=self.weights_out[idx][i][j])
                    # self.A[i + offset_in][j + offset_ou] = self.weights_out[idx][i][j]
        self.graph = G
        # self.graph = self.delet_G_edge(self.graph, threshold)

        self.node_feature = self.Cal_feature(node_fea=node_fea)    # 计算图节点特征
        self.node_feature = np.array(self.node_feature).reshape(1, -1)
        self.A = np.array(nx.adjacency_matrix(self.graph).todense())    # 邻接矩阵
        self.node_feature = np.dot(self.node_feature, self.A)
        # feature = self.Cal_feature2(self.node_feature.reshape(-1)).reshape(-1)
        feature = self.node_feature

        # if node_fea == "DC" or node_fea == "D":
        #     self.node_feature = np.sum(self.A, axis=0).reshape(1, -1) / self.node_num  # 节点度
        #     self.node_feature = np.dot(self.node_feature, self.A)  # 将邻接矩阵和节点向量进行矩阵运算 性能提升
        #     self.node_feature = normalization(self.node_feature)
        # elif node_fea == "EC":
        #     eigenvalue, featurevector = np.linalg.eig(self.A)
        #     max_eigenvalue_idx = np.argmax(eigenvalue)
        #     self.node_feature = featurevector[max_eigenvalue_idx]
        #     self.node_feature = np.dot(self.node_feature, self.A)  # 将邻接矩阵和节点向量进行矩阵运算 性能提升
        #     self.node_feature = normalization(self.node_feature)
        # feature = self.node_feature

        return G, self.A, feature.reshape(-1)

    def Generate_act(self, sample):
        self.Cal_layers_out(sample)
        # print(self.layers_out[-2][0], self.layers_out[-1][0])
        layer_out = np.concatenate((self.layers_out[-2][0], self.layers_out[-1][0]), axis=0)
        # layer_out = np.concatenate((self.Cal_vector_statistic(self.layers_out[-2][0]),
        #                             self.Cal_vector_statistic(self.layers_out[-1][0])), axis=0)
        return layer_out

    def Cal_feature(self, node_fea="EC"):
        num = self.node_num
        # 计算图指标
        feature = [0]*num
        if node_fea == "EC":
            iter = 1000
            while True: # 迭代次数不够多会引起报错
                try:
                    # 特征向量中心度
                    EC = nx.eigenvector_centrality(self.graph, max_iter=iter, weight="weight")  # 字典格式
                    break
                except:
                    iter += 1000
                    if iter > 10000:    # 防止一直算不出卡在这
                        EC = np.zeros(num)
                        break
            for i in range(num):
                try:
                    feature[i] = EC[i]
                except:     # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "CC":
            CC = nx.closeness_centrality(self.graph)  # 紧密中心度
            for i in range(num):
                try:
                    feature[i] = CC[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "D":
            D = nx.degree(self.graph)  # 度
            for i in range(num):
                try:
                    feature[i] = D[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "DC":
            DC = nx.degree_centrality(self.graph)  # 度中心度
            for i in range(num):
                try:
                    feature[i] = DC[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue
        elif node_fea == "BC":
            BC = nx.betweenness_centrality(self.graph)  # 间接中心性
            for i in range(num):
                try:
                    feature[i] = BC[i]
                except:  # 部分节点被删除
                    feature[i] = 0
                    continue

        return feature

    def Cal_feature2(self, node_feature):
        # 将节点特征进一步提取统计特征
        feature_1 = node_feature[-10:]  # 最后一层节点度特征
        feature_1 = self.Cal_vector_statistic(feature_1)
        feature_2 = node_feature[-266:-10]
        feature_2 = self.Cal_vector_statistic(feature_2)
        feature = np.concatenate((feature_1, feature_2), axis=0)
        return feature

    def Cal_vector_statistic(self, vector):
        # 计算向量的统计特征
        s = pd.Series(vector)
        feature = np.array([np.max(vector), np.mean(vector), np.median(vector), np.std(vector), s.skew(), s.kurt()])
        feature = normalization(feature).reshape(-1)
        return feature

    def Draw_graph(self, save_path=None, show=True):
        plt.close()
        # plt.rcParams['figure.figsize'] = (10, 10)
        pos = {}
        filter = np.array(self.filters_shape)
        max_filter = np.max(filter)  # 某一层最大的神经元数
        n_color = [(217/255,209/255,155/255), (153/255,164/255,188/255), (101/255,102/255,103/255),
                   (150/255,80/255,75/255), (164/255,152/255,133/255), (215/255,202/255,177/255)]    # 每层神经元的颜色
        node_color = []
        # 设置每个神经元的位置
        for layer in range(filter.size):  # 层号
            base_idx = np.sum(filter[:layer])  # 每层号的base
            # base_y = int(max_filter/2)-int(filter[layer]/2)
            base_y = 0
            jiange = (max_filter / (filter[layer] - 1 + 1e-5))  # 同层节点之间的距离
            for idx in range(filter[layer]):  # 每层的号
                x = layer
                # y = base_y + idx
                y = base_y + int(idx * jiange)
                pos[base_idx + idx] = (x, y)
                node_color.append(n_color[layer])

        # node_size = [v * 30 for v in dict(self.graph.degree).values()]
        node_size = [v * 700+40 for v in self.node_feature.reshape(-1)]
        edge_color = [float(d['weight']) for (u, v, d) in self.graph.edges(data=True)]

        nodes = nx.draw_networkx_nodes(self.graph, pos,
                                       node_size=node_size)
        edges = nx.draw_networkx_edges(self.graph, pos, edge_color="#c0c0c0", width=2)
        # edges = nx.draw_networkx_edges(self.graph, pos, edge_color=edge_color, width=2,
        #                                edge_cmap=plt.cm.cividis)

        # nx.draw(self.graph, pos=pos, with_labels=False,
        #         # node_size=node_size,
        #         edge_color=edge_color, width=2)
        # plt.colorbar(edges)
        # plt.axis('off')
        plt.tight_layout()
        if save_path!=None:
            plt.savefig(save_path)
        if show:
            plt.show()

    def delet_G_edge(self, G, threshold=0):
        # 删除权重小于等于threshold的连边
        for i in list(G.edges()):
            # print(i)
            weight = G[i[0]][i[1]]["weight"]
            if weight <= threshold:  # 如果权边权重小于阈值 ，删除
                G.remove_edge(i[0], i[1])
        return G

    def delet_G_node(self, G):
        # 删除没有连边的节点
        for i in list(G.adj.keys()):
            # print(i)
            if G.adj[i] == {}:  # 节点无连边
                G.remove_node(i)
        return G

    def delet_G_other_node(self, G, min, max):
        # 删除无输入节点
        for node_idx in list(G.adj.keys()):
            if node_idx >= min:
                if (np.array(list(G.adj[node_idx].keys())) > node_idx).all():  # 该图节点不存在输入
                    for next_node_idx in list(G.adj[node_idx].keys()):
                        G.remove_edge(node_idx, next_node_idx)
                    # G.remove_node(node_idx)
        # 删除无输出节点
        for node_idx in list(G.adj.keys()):
            if node_idx < max:
                if (np.array(list(G.adj[node_idx].keys())) < node_idx).all():  # 该图节点不存在输出
                    for next_node_idx in list(G.adj[node_idx].keys()):
                        G.remove_edge(node_idx, next_node_idx)
                    # G.remove_node(node_idx)
        return G

    def get_layer_A(self, graph_layer_num=4):
        # 按层来获取邻接矩阵
        if type(self.A) == type(None):
            print("未用样本激活神经元")
            return self.A
        in_beg = np.sum(self.filters_shape[:graph_layer_num])
        in_end = np.sum(self.filters_shape[:graph_layer_num+1])
        out_beg = in_end
        out_end = np.sum(self.filters_shape[:graph_layer_num+2])
        A = self.A[in_beg:in_end][..., out_beg:out_end]     # 某一层的邻接矩阵
        weight_num = self.weights_index[graph_layer_num]  # 权重层的索引

        return weight_num, A

    def get_layer_node(self, graph_layer_num=4):
        # 获得
        beg = np.sum(self.filters_shape[:graph_layer_num])
        end = np.sum(self.filters_shape[:graph_layer_num+1])
        return self.node_feature[..., beg:end]


if __name__ == "__main__":
    from Load_dataset import *
    # 加载数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10()
    # 加载模型
    model = lenet5_cifar10(path="Model/Cifar10/Lenet5/Wed-Feb-23-20:33:29-2022.h5")
    model_weights_names = [weight.name for layer in model.layers for weight in layer.weights]
    model_weights = model.get_weights()
    weights_index = [0, 2, 4, 6, 8]  # weights中重要的层的索引
    # conv_index = [0, 2, 4]     # kernel中卷积层的索引
    # dens_index = [6, 8]  # kernel中全连接层的索引
    filters_shape = [3, 6, 16, 120, 84, 10]  # 重要层的filter数目
    edge_weights = [np.zeros(shape=(filters_shape[i - 1], filters_shape[i]))
                    for i in range(1, len(filters_shape))]  # 权边初始化
    weights_out = [np.zeros(shape=(filters_shape[i - 1], filters_shape[i]))
                   for i in range(1, len(filters_shape))]  # 将输出和权边相乘
    layer_out = [np.zeros(shape=(1, filters_shape[i]))
                 for i in range(1, len(filters_shape))]  # 每层输出初始化
    edge_num = sum([np.size(ary) for ary in edge_weights])  # 权边数
    node_num = sum(filters_shape)   # 节点数

    # 计算filter之间的连边权值
    for idx, edge_weight in enumerate(edge_weights):  # weights的第几层
        # print(edge_weight)
        inshape = edge_weight.shape[0]
        oushape = edge_weight.shape[1]
        if idx == 0 or idx == 1 or idx == 2:  # 卷积层的filter连边权重
            for i in range(inshape):
                for j in range(oushape):
                    edge_weights[idx][i][j] = np.linalg.norm(model_weights[weights_index[idx]][..., i, j])
        elif idx == 3:  # 卷积和全连接的连边权重
            for i in range(inshape):
                for j in range(oushape):
                    # 从卷积层到全连接层的转换，多维数组被整为一维数组，默认数据按卷积核数一一对应;（16*5*5-->1*400）
                    # 5*5是对应卷积核大小
                    # print(i, i*(5*5), (i+1)*(5*5), j)
                    edge_weights[idx][i][j] = np.linalg.norm(
                        model_weights[weights_index[idx]][..., i * (5 * 5):(i + 1) * (5 * 5), j])
        elif idx == 4:  # 全连接和全连接的连边权重
            for i in range(inshape):
                for j in range(oushape):
                    edge_weights[idx][i][j] = model_weights[weights_index[idx]][i, j]

    # 计算模型的每层输出
    get_conv_1st_output = function(inputs=[model.layers[0].input], outputs=[model.layers[2].output])
    get_conv_2st_output = function(inputs=[model.layers[0].input], outputs=[model.layers[4].output])
    get_conv_3st_output = function(inputs=[model.layers[0].input], outputs=[model.layers[6].output])
    get_dens_1st_output = function(inputs=[model.layers[0].input], outputs=[model.layers[8].output])
    get_dens_2st_output = function(inputs=[model.layers[0].input], outputs=[model.layers[9].output])
    conv_1st_output = get_conv_1st_output([X_test[0:1].tolist()])[0]
    conv_2st_output = get_conv_2st_output([X_test[0:1].tolist()])[0]
    conv_3st_output = get_conv_3st_output([X_test[0:1].tolist()])[0]
    dens_1st_output = get_dens_1st_output([X_test[0:1].tolist()])[0]
    dens_2st_output = get_dens_2st_output([X_test[0:1].tolist()])[0]

    # 对模型每层输出取范数
    for idx, edge_weight in enumerate(edge_weights):
        # inshape = edge_weight.shape[0]
        oushape = edge_weight.shape[1]
        if idx == 0:
            for i in range(oushape):
                layer_out[idx][0][i] = np.linalg.norm(conv_1st_output[..., i])
        elif idx == 1:
            for i in range(oushape):
                layer_out[idx][0][i] = np.linalg.norm(conv_2st_output[..., i])
        elif idx == 2:
            for i in range(oushape):
                layer_out[idx][0][i] = np.linalg.norm(conv_3st_output[..., i])
        elif idx == 3:
            for i in range(oushape):
                layer_out[idx][0][i] = np.linalg.norm(dens_1st_output[..., i])
        elif idx == 4:
            for i in range(oushape):
                layer_out[idx][0][i] = np.linalg.norm(dens_2st_output[..., i])
        # 每层的输出做一次归一化
        layer_out[idx] = (layer_out[idx] - np.min(layer_out[idx])) / (
                          np.max(layer_out[idx]) - np.min(layer_out[idx]))
    # 每层输出反映射到权重
    for i in range(5):
        weights_out[i] = np.multiply(edge_weights[i], layer_out[i])

    print("end")

