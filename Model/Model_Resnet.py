from keras import layers
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import Model
from keras.regularizers import l2
from keras.backend import function, gradients
import numpy as np
from common import *
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def conv2d_bn(x, filters, kernel_size, name, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay),
                   name=name+"_conv2d"
                   )(x)
    layer = BatchNormalization(name=name+"_batnorm")(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, name, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, name+"_bn", weight_decay, strides)
    layer = Activation('relu', name=name+"_relu")(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay, name, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2, name=name+"_bn1")
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              name=name+"_bnr"
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         name=name+"_bn2"
                         )
    out = layers.add([residual_x, residual], name=name+"_add")
    out = Activation('relu', name=name+"_relu")(out)
    return out


def ResNet18(classes, input_shape, weight_decay=1e-4):
    input = Input(shape=input_shape)
    x = input
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1), name="block1")

    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, name="block21")
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, name="block22")
    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True, name="block31")
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, name="block32")
    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True, name="block41")
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, name="block42")
    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True, name="block51")
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False, name="block52")
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name="fc1")(x)
    model = Model(input, x, name='ResNet18')
    return model


def ResNetForCIFAR10(classes, name, input_shape, block_layers_num, weight_decay):
    input = Input(shape=input_shape)
    x = input
    x = conv2d_bn_relu(x, filters=16, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    for i in range(block_layers_num):
        x = ResidualBlock(x, filters=16, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=32, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    for i in range(block_layers_num - 1):
        x = ResidualBlock(x, filters=32, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    for i in range(block_layers_num - 1):
        x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(8, 8), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input, x, name=name)
    return model


def ResNet20ForCIFAR10(classes, input_shape, weight_decay):
    return ResNetForCIFAR10(classes, 'resnet20', input_shape, 3, weight_decay)


def ResNet32ForCIFAR10(classes, input_shape, weight_decay):
    return ResNetForCIFAR10(classes, 'resnet32', input_shape, 5, weight_decay)


def ResNet56ForCIFAR10(classes, input_shape, weight_decay):
    return ResNetForCIFAR10(classes, 'resnet56', input_shape, 9, weight_decay)


class resnet18_cifar10_graph():
    def __init__(self, model):
        # 读取模型权重和权重名
        self.model = model
        self.model_weights = self.model.get_weights()
        self.model_weights_names = [weight.name for layer in self.model.layers for weight in layer.weights]

        # 构图初始化
        self.weights_index = [70, 75, 90, 95, 100]  # weights中重要的层的索引
        self.filters_shape = [256, 256, 512, 512, 512, 10]  # 重要层的filter数目
        self.conv_idx = [0, 1, 2, 3]  # 在self.weights_index中属于卷积层的索引
        # self.conv_fc_idx = [2]
        self.fc_idx = [4]

        # self.weights_index = [70, 75, 90, 95, 100]  # weights中重要的层的索引
        # self.filters_shape = [256, 256, 256, 512, 512, 10]  # 重要层的filter数目
        # self.conv_idx = [0, 1, 2, 3]  # 在self.weights_index中属于卷积层的索引
        # # self.conv_fc_idx = [2]
        # self.fc_idx = [4]

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
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[47].output]),  # 47
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[51].output]),  # 51
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[60].output]),  # 60
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[63].output]),  # 63
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[68].output])
        ]
        # self.get_layers_fuc = [
        #     function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[41].output]),  #
        #     function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[48].output]),  #
        #     function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[57].output]),  #
        #     function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[64].output]),  #
        #     function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[68].output])
        # ]

    def Cal_weights_edges(self):
        # 计算filter之间的连边值
        for idx, edge_weight in enumerate(self.edge_weights):  # weights的第几层
            inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx in self.conv_idx:  # 卷积层
                self.edge_weights[idx] = np.linalg.norm(self.model_weights[self.weights_index[idx]], axis=(0, 1))
                # self.edge_weights[idx] = 1
            elif idx in self.fc_idx:  # 全连接层
                self.edge_weights[idx] = self.model_weights[self.weights_index[idx]]
                # self.edge_weights[idx] = 1
            # 对每层权重做 归一化
            self.edge_weights[idx] = normalization(normalization(self.edge_weights[idx]))
            # 对每层权重做 标准化
            # self.edge_weights[idx] = standardization(self.edge_weights[idx])
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
        self.A_T = self.A.T
        # neigh_num = np.sum(self.A > 0, axis=0)
        self.node_feature1 = np.sum(self.A, axis=0).reshape(1, -1) / self.node_num  # 节点度
        self.node_feature1 = np.dot(self.node_feature1, self.A)  # 将邻接矩阵和节点向量进行矩阵运算 性能提升
        self.node_feature1 = normalization(self.node_feature1)
        # feature = self.node_feature1.reshape(-1)
        feature = self.Cal_feature(self.node_feature1.reshape(-1))

        # self.A_T = self.A.transpose()  # 转置
        # self.node_feature1 = np.sum(self.A, axis=0).reshape(1, -1) / self.node_num  # 节点度
        # self.node_feature1 = np.dot(np.dot(self.node_feature1, self.A_T), self.A)
        # feature = self.Cal_feature(self.node_feature1.reshape(-1))

        return G, self.A, feature

    def Generate_act(self, sample):
        self.Cal_layers_out(sample)
        layer_out = np.concatenate((self.layers_out[-2][0], self.layers_out[-1][0]), axis=0)    # 将置信度层和隐层特征组合
        # layer_out = self.layers_out[-1]  # 将置信度层和隐层特征组合
        return layer_out

    def Cal_feature(self, node_feature):
        # 将节点特征进一步提取统计特征
        feature_1 = node_feature[-10:]  # 最后一层节点度特征
        feature_1 = self.Cal_vector_statistic(feature_1)
        feature_2 = node_feature[-522:-10]
        feature_2 = self.Cal_vector_statistic(feature_2)
        feature = np.concatenate((feature_1, feature_2), axis=0)

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
        # 获得特定层的节点特征
        beg = np.sum(self.filters_shape[:graph_layer_num])
        end = np.sum(self.filters_shape[:graph_layer_num+1])
        # print("beg, end", beg, end)
        return self.node_feature1[..., beg:end]

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


class resnet18_cif10_resn_graph():
    def __init__(self, model):
        # 读取模型权重和权重名
        self.model = model
        self.model_weights = self.model.get_weights()
        self.model_weights_names = [weight.name for layer in self.model.layers for weight in layer.weights]

        # 构图初始化
        self.weights_index = [70, 75, 90, 95, 100]  # weights中重要的层的索引
        self.filters_shape = [256, 256, 512, 512, 512, 10]  # 重要层的filter数目
        self.conv_idx = [0, 1, 2, 3]  # 在self.weights_index中属于卷积层的索引
        # self.conv_fc_idx = [2]
        self.fc_idx = [4]
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
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[0].output]),  #
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[55].output]),  #
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[57].output]),  # shortcut
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[64].output]),  # add
            function(inputs=[self.model.layers[0].input], outputs=[self.model.layers[68].output])
        ]

    def Cal_weights_edges(self):
        # 计算filter之间的连边值
        for idx, edge_weight in enumerate(self.edge_weights):  # weights的第几层
            inshape = edge_weight.shape[0]
            oushape = edge_weight.shape[1]
            if idx in self.conv_idx:  # 卷积层
                self.edge_weights[idx] = np.linalg.norm(self.model_weights[self.weights_index[idx]], axis=(0, 1))
            elif idx in self.fc_idx:  # 全连接层
                self.edge_weights[idx] = self.model_weights[self.weights_index[idx]]
            # 对每层权重做 归一化
            self.edge_weights[idx] = normalization(normalization(self.edge_weights[idx]))
            # 对每层权重做 标准化
            # self.edge_weights[idx] = standardization(self.edge_weights[idx])
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
        self.A_T = self.A
        self.node_feature1 = np.sum(self.A, axis=0).reshape(1, -1) / self.node_num  # 节点度
        self.node_feature1 = np.dot(self.node_feature1, self.A)  # 将邻接矩阵和节点向量进行矩阵运算 性能提升
        # self.node_feature = normalization(self.node_feature)
        # feature = self.node_feature.reshape(-1)
        feature = self.Cal_feature(self.node_feature1.reshape(-1))

        return G, self.A, feature

    def Generate_act(self, sample):
        self.Cal_layers_out(sample)
        layer_out = np.concatenate((self.layers_out[-2][0], self.layers_out[-1][0]), axis=0)  # 将置信度层和隐层特征组合
        # layer_out = self.layers_out[-1]  # 将置信度层和隐层特征组合
        return layer_out

    def Cal_feature(self, node_feature):
        # 将节点特征进一步提取统计特征
        feature_1 = node_feature[-10:]  # 最后一层节点度特征
        feature_1 = self.Cal_vector_statistic(feature_1)
        feature_2 = node_feature[-522:-10]
        feature_2 = self.Cal_vector_statistic(feature_2)
        feature = np.concatenate((feature_1, feature_2), axis=0)

        return feature

    def Cal_vector_statistic(self, vector):
        # 计算向量的统计特征
        s = pd.Series(vector)
        feature = np.array([np.max(vector), np.mean(vector), np.median(vector), np.std(vector), s.skew(), s.kurt()])
        # feature = normalization(feature).reshape(-1)
        return feature

    def delet_G_edge(self, A, threshold=0):
        # A为上三角邻接矩阵
        A = np.where(A > threshold, A, 0)
        return A

    def get_layer_A(self, graph_layer_num=3):
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
        # 获得特定层的节点特征
        beg = np.sum(self.filters_shape[:graph_layer_num])
        end = np.sum(self.filters_shape[:graph_layer_num + 1])
        # print("beg, end", beg, end)
        return self.node_feature[..., beg:end]


if __name__ == "__main__":

    print("end")


