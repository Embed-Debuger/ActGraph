# import tensorflow.compat.v1 as tf
from keras import optimizers, backend, losses, regularizers
from keras.backend import function, gradients
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, ReLU, MaxPool2D
from keras.models import Model

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from common import *
import taichi as ti

def conv2d_bn_relu(model,
                   filters,
                   block_index, layer_index,
                   weight_decay=.0, padding='same'):
    conv_name = 'conv' + str(block_index) + '-' + str(layer_index)
    model = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   padding=padding,
                   kernel_regularizer=regularizers.l2(weight_decay),
                   strides=(1, 1),
                   name=conv_name,
                   )(model)
    bn_name = 'bn' + str(block_index) + '-' + str(layer_index)
    model = BatchNormalization(name=bn_name)(model)
    relu_name = 'relu' + str(block_index) + '-' + str(layer_index)
    model = Activation('relu', name=relu_name)(model)
    return model


def dense2d_bn_dropout(model, units, weight_decay, name):
    model = Dense(units,
                  kernel_regularizer=regularizers.l2(weight_decay),
                  name=name,
                  )(model)
    model = BatchNormalization(name=name + '-bn')(model)
    model = Activation('relu', name=name + '-relu')(model)
    model = Dropout(0.5, name=name + '-dropout')(model)
    return model

# https://github.com/jerett/Keras-CIFAR10/blob/master
def VGGNet(classes, input_shape, weight_decay,
           conv_block_num=5,
           fc_layers=2, fc_units=4096):
    input = Input(shape=input_shape)
    # block 1
    x = conv2d_bn_relu(model=input,
                       filters=64,
                       block_index=1, layer_index=1,
                       weight_decay=weight_decay
                       )
    x = conv2d_bn_relu(model=x,
                       filters=64,
                       block_index=1, layer_index=2,
                       weight_decay=weight_decay)
    x = MaxPool2D(name='pool1')(x)

    # block 2
    if conv_block_num >= 2:
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=2,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool2')(x)

    # block 3
    if conv_block_num >= 3:
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool3')(x)

    # block 4
    if conv_block_num >= 4:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool4')(x)

    # block 5
    if conv_block_num >= 5:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool5')(x)

    x = Flatten(name='flatten')(x)
    if fc_layers >= 1:
        x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc1')
        if fc_layers >= 2:
            x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc2')
    out = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(input, out)
    return model

def VGG16(classes):
    return VGGNet(classes, weight_decay=1e-4, conv_block_num=5, fc_layers=2, fc_units=512)


def VGG19Net(classes, input_shape, weight_decay,
           conv_block_num=5,
           fc_layers=2, fc_units=4096):
    input = Input(shape=input_shape)
    # block 1
    x = conv2d_bn_relu(model=input,
                       filters=64,
                       block_index=1, layer_index=1,
                       weight_decay=weight_decay
                       )
    x = conv2d_bn_relu(model=x,
                       filters=64,
                       block_index=1, layer_index=2,
                       weight_decay=weight_decay)
    x = MaxPool2D(name='pool1')(x)

    # block 2
    if conv_block_num >= 2:
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=2,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool2')(x)

    # block 3
    if conv_block_num >= 3:
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=3,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=4,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool3')(x)

    # block 4
    if conv_block_num >= 4:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=3,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=4,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool4')(x)

    # block 5
    if conv_block_num >= 5:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=3,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=4,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool5')(x)

    x = Flatten(name='flatten')(x)
    if fc_layers >= 1:
        x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc1')
        if fc_layers >= 2:
            x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc2')
    out = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(input, out)
    return model


class vgg16_graph():
    def __init__(self, model, Dataset="Cifar10"):
        # 读取模型权重和权重名
        self.model = model
        self.model_weights = self.model.get_weights()
        self.model_weights_names = [weight.name for layer in self.model.layers for weight in layer.weights]

        # 构图初始化
        self.weights_index = [42, 54, 60, 66, 72]  # weights中重要的层的索引
        self.filters_shape = [256, 512, 512, 256, 256, 10]  # 重要层的filter数目
        self.conv_idx = [0, 1]     # 在self.weights_index中属于卷积层的索引
        self.conv_fc_idx = [2]
        self.fc_idx = [3, 4]
        # self.weights_index = [66, 72]  # weights中重要的层的索引
        # self.filters_shape = [256, 256, 10]  # 重要层的filter数目
        self.edge_weights = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                             for i in range(1, len(self.filters_shape))]  # 权边初始化
        self.weights_out = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                            for i in range(1, len(self.filters_shape))]  # 将输出和权边相乘（和模型权重及其输出有关）
        self.layers_out = [np.zeros(shape=(1, self.filters_shape[i]))
                           for i in range(1, len(self.filters_shape))]  # 每层输出初始化
        self.layers_grad_out = [np.zeros(shape=(1, self.filters_shape[i]))
                                for i in range(1, len(self.filters_shape))]  # 每层输出初始化
        self.edge_num = sum([np.size(ary) for ary in self.edge_weights])  # 权边数
        self.node_num = sum(self.filters_shape)  # 结点数
        self.edge_weights = self.Cal_weights_edges()  # 计算有权连边(仅和模型权重相关)
        self.graph = None  # 图
        self.node_feature = None    # 节点特征（度）
        self.A = np.zeros(shape=(self.node_num, self.node_num))     # 邻接矩阵

        # 初始化模型重要层输出 模型太大 只对最后几层构图
        self.get_layers_fuc = [
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[26].output]),
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[32].output]),
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[37].output]),
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[41].output]),
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[44].output])
        ]

        # self.pred = self.model.output
        # self.true = backend.argmax(self.model.output)
        # self.loss = losses.sparse_categorical_crossentropy(y_true=self.true[0], y_pred=self.pred)
        # self.get_layers_grad = [
        #     gradients(self.loss, self.model.layers[26].input)[0][0],
        #     gradients(self.loss, self.model.layers[32].input)[0][0],
        #     gradients(self.loss, self.model.layers[37].input)[0][0],
        #     gradients(self.loss, self.model.layers[41].input)[0][0],
        #     gradients(self.loss, self.model.layers[44].input)[0][0]
        # ]

    def Cal_weights_edges(self):
        # 计算filter之间的连边值
        for idx, edge_weight in enumerate(self.edge_weights):  # weights的第几层
            inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx in self.conv_idx:    # 卷积层
                self.edge_weights[idx] = np.linalg.norm(self.model_weights[self.weights_index[idx]], axis=(0, 1))
            if idx in self.conv_fc_idx:  # flatten-fc
                for i in range(inshape):
                    for j in range(oushape):
                        # 从卷积层到全连接层的转换，多维数组被整为一维数组，默认数据按卷积核数一一对应;（16*5*5-->1*400）
                        # 2*2是对应卷积核大小
                        self.edge_weights[idx][i][j] = np.linalg.norm(
                            self.model_weights[self.weights_index[idx]][i * (2 * 2):(i + 1) * (2 * 2), j])
            elif idx in self.fc_idx:   # 全连接层
                self.edge_weights[idx] = self.model_weights[self.weights_index[idx]]

            # 对每层权重做 归一化
            self.edge_weights[idx] = normalization(normalization(self.edge_weights[idx]))
            # 对每层权重做 标准化
            # self.edge_weights[idx] = self.standardization(self.edge_weights[idx])
        return self.edge_weights

    def Cal_layers_out(self, sample):
        # 计算每层的filter输出
        # 将filter的输出映射到权重
        layers_act = [func([sample.tolist()])[0] for func in self.get_layers_fuc]   # 样本激活每层输出
        # 对模型每层输出取范数
        for idx, edge_weight in enumerate(self.edge_weights):
            # inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            for i in range(oushape):
                self.layers_out[idx][0][i] = np.linalg.norm(layers_act[idx][..., i])
            # 每层的输出做一次归一化
            self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
        self.Cal_out_mul_weights()  # 计算模型输出映射到权重的值

        # sess = backend.get_session()
        # layers_grad = sess.run([fun for fun in self.get_layers_grad],
        #                        # feed_dict={self.model.input: sample, self.true: label})
        #                        feed_dict={self.model.input: sample})  # 注意这里传递参数的情况
        #
        # # layers_grad = [func([sample.tolist()])[0] for func in self.get_layers_grad] # 样本每层的梯度
        # for idx, edge_weight in enumerate(self.edge_weights):
        #     # inshape = edge_weight.shape[0]
        #     oushape = edge_weight.shape[1]
        #     for i in range(oushape):
        #         self.layers_grad_out[idx][0][i] = np.linalg.norm(layers_grad[idx][..., i])
        #     # 每层的梯度做一次归一化
        #     self.layers_grad_out[idx] = normalization(self.layers_grad_out[idx]) * 1

    def Cal_out_mul_weights(self):
        # 每层输出反映射到权重
        for i in range(len(self.weights_index)):    # 输出反乘权重
            self.weights_out[i] = np.multiply(self.edge_weights[i], self.layers_out[i])
        # for i in range(len(self.weights_index)):    # 输入乘权重
        #     self.weights_out[i+1] = np.multiply(self.layers_out[i], self.edge_weights[i+1].T).T
        #     if i == len(self.weights_index)-2:
        #         break

    def Generate_graph(self, sample, node_fea="DC", threshold=0):
        self.Cal_layers_out(sample)
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
                    # print(i, j)
                    # G.add_edge(i + offset_in, j + offset_ou, weight=self.weights_out[idx][i][j])
                    self.A[i + offset_in][j + offset_ou] = self.weights_out[idx][i][j]

        # G = self.delet_G_edge(G, threshold=threshold)  # 删减连边
        # G = self.delet_G_node(G)     # 删减多余节点
        # G = self.delet_G_other_node(G, min=self.filters_shape[0], max=np.sum(self.filters_shape[:-1])) # 删除无输入或者无输出节点
        # self.graph = G
        # self.node_feature = self.Cal_feature(node_fea=node_fea)  # 计算图节点特征
        # self.node_feature = np.array(self.node_feature).reshape(1, -1)
        # self.A = np.array(nx.adjacency_matrix(self.graph).todense())  # 邻接矩阵
        # feature = np.dot(self.node_feature, self.A)
        # return G, self.A, feature.reshape(-1)

        self.node_feature = np.sum(self.A, axis=0).reshape(1, -1)   # 节点度
        self.node_feature = np.dot(self.node_feature, self.A)   # 将邻接矩阵和节点向量进行矩阵运算 性能提升
        self.node_feature = normalization(self.node_feature)
        # feature = self.node_feature.reshape(-1)
        feature = self.Cal_feature(self.node_feature.reshape(-1))
        return G, self.A, feature


    def Generate_act(self, sample):
        self.Cal_layers_out(sample)
        # layer_out = np.concatenate((self.layers_out[-2][0], self.layers_out[-1][0]), axis=0)    # 将置信度层和隐层特征组合
        layer_out = np.concatenate((self.Cal_vector_statistic(self.layers_out[-2][0]),
                                    self.Cal_vector_statistic(self.layers_out[-1][0])), axis=0)
        return layer_out

    def Cal_feature(self, node_feature):
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
        plt.rcParams['figure.figsize'] = (10, 10)
        pos = {}
        filter = np.array(self.filters_shape)
        max_filter = np.max(filter)  # 某一层最大的神经元数
        # n_color = [(217/255,209/255,155/255), (153/255,164/255,188/255), (101/255,102/255,103/255),
        #            (150/255,80/255,75/255), (164/255,152/255,133/255), (215/255,202/255,177/255)]    # 每层神经元的颜色
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
                # node_color.append(n_color[layer])

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

    def delet_G_edge2(self, A, threshold=0):
        # A为上三角邻接矩阵
        A = np.where(A > threshold, A, 0)
        return A

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

    def get_layer_A(self, graph_layer_num=3):
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

    def get_layer_node(self, graph_layer_num=3):
        # 获得
        beg = np.sum(self.filters_shape[:graph_layer_num])
        end = np.sum(self.filters_shape[:graph_layer_num+1])
        # print("beg, end", beg, end)
        return self.node_feature[..., beg:end]


class vgg19_cifar100_graph():
    def __init__(self, model):
        # 读取模型权重和权重名
        self.model = model
        self.model_weights = self.model.get_weights()
        self.model_weights_names = [weight.name for layer in self.model.layers for weight in layer.weights]

        # 构图初始化
        self.weights_index = [60, 66, 72, 78, 84]  # weights中重要的层的索引
        self.filters_shape = [512, 512, 512, 256, 256, 100]  # 重要层的filter数目
        self.conv_idx = [0, 1]  # 在self.weights_index中属于卷积层的索引
        self.conv_fc_idx = [2]
        self.fc_idx = [3, 4]
        self.edge_weights = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                             for i in range(1, len(self.filters_shape))]  # 权边初始化
        self.weights_out = [np.zeros(shape=(self.filters_shape[i - 1], self.filters_shape[i]))
                            for i in range(1, len(self.filters_shape))]  # 将输出和权边相乘（和模型权重及其输出有关）
        self.layers_out = [np.zeros(shape=(1, self.filters_shape[i]))
                           for i in range(1, len(self.filters_shape))]  # 每层输出初始化
        self.edge_num = sum([np.size(ary) for ary in self.edge_weights])  # 权边数
        self.node_num = sum(self.filters_shape)  # 结点数
        self.edge_weights = self.Cal_weights_edges()  # 计算有权连边(仅和模型权重相关)
        self.graph = None  # 图
        self.A = np.zeros(shape=(self.node_num, self.node_num))  # 邻接矩阵上三角
        self.A_T = np.zeros(shape=(self.node_num, self.node_num))  # 邻接矩阵下三角
        self.node_feature1 = None  # 节点特征（入度）
        self.node_feature2 = None  # 节点特征（出度）

        # 初始化模型重要层输出 模型太大 只对最后几层构图
        self.get_layers_fuc = [
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[35].output]),  # conv
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[38].output]),  # conv
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[43].output]),  # fc
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[47].output]),  # fc
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[50].output])   # fc
        ]

    def Cal_weights_edges(self):
        # 计算filter之间的连边值
        for idx, edge_weight in enumerate(self.edge_weights):  # weights的第几层
            inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx in self.conv_idx:  # 卷积层
                self.edge_weights[idx] = np.linalg.norm(self.model_weights[self.weights_index[idx]], axis=(0, 1))
            if idx in self.conv_fc_idx:  # flatten-fc
                for i in range(inshape):
                    for j in range(oushape):
                        # 从卷积层到全连接层的转换，多维数组被整为一维数组，默认数据按卷积核数一一对应;（16*5*5-->1*400）
                        # 2*2是对应卷积核大小
                        self.edge_weights[idx][i][j] = np.linalg.norm(
                            self.model_weights[self.weights_index[idx]][i * (2 * 2):(i + 1) * (2 * 2), j])
            elif idx in self.fc_idx:  # 全连接层
                self.edge_weights[idx] = self.model_weights[self.weights_index[idx]]
            # 对每层权重做 归一化
            self.edge_weights[idx] = normalization(normalization(self.edge_weights[idx]))
            # 对每层权重做 标准化
            # self.edge_weights[idx] = self.standardization(self.edge_weights[idx])
        return self.edge_weights

    def Cal_layers_out(self, sample):
        # 计算每层的filter输出
        # 将filter的输出映射到权重
        layers_act = [func([sample.tolist()])[0] for func in self.get_layers_fuc]  # 样本激活每层输出
        # 对模型每层输出取范数
        for idx, edge_weight in enumerate(self.edge_weights):
            # inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            for i in range(oushape):
                self.layers_out[idx][0][i] = np.linalg.norm(layers_act[idx][..., i])
            # 每层的输出做一次归一化
            self.layers_out[idx] = normalization(normalization(self.layers_out[idx])) * 1
        self.Cal_out_mul_weights()  # 计算模型输出映射到权重的值

    def Cal_out_mul_weights(self):
        # 每层输出反映射到权重
        for i in range(len(self.weights_index)):
            self.weights_out[i] = np.multiply(self.edge_weights[i], self.layers_out[i])

    def Generate_graph(self, sample, node_fea="DC", threshold=0):
        self.Cal_layers_out(sample)
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
                    self.A[i + offset_in][j + offset_ou] = self.weights_out[idx][i][j]
                    # G.add_edge(i + offset_in, j + offset_ou, weight=self.weights_out[idx][i][j])
        self.A_T = self.A.transpose()   # 转置
        self.node_feature1 = np.sum(self.A, axis=0).reshape(1, -1) / self.node_num  # 节点度
        self.node_feature1 = np.dot(self.node_feature1, self.A)  # 将邻接矩阵和节点向量进行矩阵运算 性能提升
        self.node_feature1 = normalization(self.node_feature1)
        # feature = self.node_feature1.reshape(-1)
        feature = self.Cal_feature(self.node_feature1.reshape(-1)).reshape(-1)

        # self.graph = G
        # self.node_feature1 = self.Cal_feature2(node_fea=node_fea)  # 计算图节点特征
        # self.node_feature1 = np.array(self.node_feature1).reshape(1, -1)
        # self.A = np.array(nx.adjacency_matrix(self.graph).todense())  # 邻接矩阵
        # feature = np.dot(self.node_feature1, self.A).reshape(-1)

        return G, self.A, feature

    def Generate_act(self, sample):
        self.Cal_layers_out(sample)
        layer_out = np.concatenate((self.layers_out[-2][0], self.layers_out[-1][0]), axis=0)  # 将置信度层和隐层特征组合

        return layer_out

    def Cal_feature(self, node_feature):
        # 将节点特征进一步提取统计特征
        feature_1 = node_feature[-100:]  # 最后一层节点度特征
        feature_1 = self.Cal_vector_statistic(feature_1)
        feature_2 = node_feature[-356:-100]
        feature_2 = self.Cal_vector_statistic(feature_2)
        feature = np.concatenate((feature_1, feature_2), axis=0)
        return feature

    def Cal_feature2(self, node_fea="EC"):
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

    def Cal_vector_statistic(self, vector):
        # 计算向量的统计特征
        s = pd.Series(vector)
        feature = np.array([np.max(vector), np.mean(vector), np.median(vector), np.std(vector), s.skew(), s.kurt()])
        feature = normalization(feature).reshape(-1)
        return feature

    def delet_G_edge(self, A, threshold=0):
        # A为上三角邻接矩阵
        A = np.where(A > threshold, A, 0)
        return A

    def get_layer_A(self, graph_layer_num):
        # 按层来获取邻接矩阵
        if type(self.A) == type(None):
            print("未用样本激活神经元")
            return self.A
        in_beg = np.sum(self.filters_shape[:graph_layer_num])
        in_end = np.sum(self.filters_shape[:graph_layer_num + 1])
        out_beg = in_end
        out_end = np.sum(self.filters_shape[:graph_layer_num + 2])
        A = self.A[in_beg:in_end][..., out_beg:out_end]  # 某一层的邻接矩阵
        weight_num = self.weights_index[graph_layer_num]  # 权重层的索引

        return weight_num, A

    def get_layer_node(self, graph_layer_num=3):
        # 获得
        beg = np.sum(self.filters_shape[:graph_layer_num])
        end = np.sum(self.filters_shape[:graph_layer_num + 1])
        # print("beg, end", beg, end)
        return self.node_feature1[..., beg:end]


if __name__ == "__main__":
    from Constant import *
    # 加载模型
    model = get_model(Dataset="Cifar10", Mo="VGG16")
    model.summary()
    # 加载数据
    X_train, Y_train, X_test, Y_test = get_dataset(Dataset="Cifar10", Mo="VGG16", Operator="origin")

    vgg16_handle = vgg16_graph(model)

    from tqdm import tqdm
    result = [vgg16_handle.Generate_graph(sample=X_test[i:i+1], node_fea="DC", threshold=0)[2] for i in tqdm(range(10))]


    # # weights_index = [0, 6, 12, 14, 20, 22, 28, 30, 36, 38, 40]  # get_weights中重要的层的索引
    # # conv_index = [0, 1, 2, 3, 4, 5, 6, 7]     # weights_index中卷积层的索引
    # # dens_index = [8, 9, 10]   # weights_index中全连接层的索引
    # # filters_shape = [3, 64, 128, 256, 256, 512, 512, 512, 512, 256, 256, 10]  # 重要层的filter数目
    # weights_index = [0, 6, 12, 14, 20, 30, 36, 38, 40]
    # conv_index = [0, 1, 2, 3, 4, 5]
    # dens_index = [6, 7, 8]
    # filters_shape = [3, 64, 128, 256, 256, 512, 512, 256, 256, 10]
    # edge_weights = [np.zeros(shape=(filters_shape[i - 1], filters_shape[i]))
    #                 for i in range(1, len(filters_shape))]  # 权边初始化
    # weights_out = [np.zeros(shape=(filters_shape[i - 1], filters_shape[i]))
    #                for i in range(1, len(filters_shape))]  # 将层输出和权边相乘
    # layer_out = [np.zeros(shape=(1, filters_shape[i]))
    #              for i in range(1, len(filters_shape))]  # 每层输出初始化
    # edge_num = sum([np.size(ary) for ary in edge_weights])  # 权边数
    # node_num = sum(filters_shape)  # 节点数
    #
    # # 计算filter之间的连边权值
    # for idx, edge_weight in enumerate(edge_weights):  # weights的第几层
    #     inshape = edge_weight.shape[0]
    #     oushape = edge_weight.shape[1]
    #     if idx in conv_index:  # 卷积层的filter连边权重
    #         for i in range(inshape):
    #             for j in range(oushape):
    #                 edge_weights[idx][i][j] = np.linalg.norm(model_weights[weights_index[idx]][..., i, j])
    #     if idx in dens_index:
    #         for i in range(inshape):
    #             for j in range(oushape):
    #                 edge_weights[idx][i][j] = model_weights[weights_index[idx]][i, j]
    # # 定义层间输出
    # get_layers_fuc = [
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[3].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[6].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[7].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[10].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[11].output]),
    #     # function(inputs=[model.layers[0].input], outputs=[model.layers[14].output]),
    #     # function(inputs=[model.layers[0].input], outputs=[model.layers[15].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[18].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[20].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[22].output]),
    #     function(inputs=[model.layers[0].input], outputs=[model.layers[24].output])]
    # layers_act = [func([X_test[0:1].tolist()])[0] for func in get_layers_fuc]   # 样本激活每层输出
    #
    # # 对模型每层输出取范数
    # for idx, edge_weight in enumerate(edge_weights):
    #     # inshape = edge_weight.shape[0]
    #     oushape = edge_weight.shape[1]
    #     for i in range(oushape):
    #         layer_out[idx][0][i] = np.linalg.norm(layers_act[idx][..., i])
    #     # 每层的输出做一次归一化
    #     layer_out[idx] = (layer_out[idx] - np.min(layer_out[idx])) / (
    #             np.max(layer_out[idx]) - np.min(layer_out[idx]))
    # # 每层输出反映射到权重
    # for i in range(len(weights_index)):
    #     weights_out[i] = np.multiply(edge_weights[i], layer_out[i])
    #
    # # from tqdm import tqdm
    # # graph_handle = vgg11_2_graph(model)
    # # result = np.array([graph_handle.Generate_graph(np.array([x])) for x in tqdm(X_test[:50])])

    print("end")

