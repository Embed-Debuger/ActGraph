from keras.models import load_model, save_model
from Model.Model_Resnet import ResNet18
from keras import optimizers
import numpy as np


if __name__ == "__main__":
    # model = ResNet18(10, (32, 32, 3))
    # model.load_weights("Model_weight/Cifar10/Resnet18/weights.h5")
    # opt = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=False)
    # model.compile(optimizer=opt,
    #               loss="sparse_categorical_crossentropy",
    #               # loss=losses.categorical_crossentropy,
    #               metrics=['accuracy'])
    # save_model(model, "Model_weight/Cifar10/Resnet18/weights.h5")
    # # 加载模型
    # model = load_model("Model_weight/Cifar10/Resnet18/weights.h5")
    # model_weights = model.get_weights()
    # model_weights_names = [weight.name for layer in model.layers for weight in layer.weights]
    #
    # weights_index = [70, 75, 90, 95, 100]  # weights中重要的层的索引
    # filters_shape = [256, 256, 512, 512, 512, 10]  # 重要层的filter数目
    # conv_idx = [0, 1, 2, 3]
    # fc_idx = [4]
    # edge_weights = [np.zeros(shape=(filters_shape[i - 1], filters_shape[i]))
    #                 for i in range(1, len(filters_shape))]  # 权边初始化
    # weights_out = [np.zeros(shape=(filters_shape[i - 1], filters_shape[i]))
    #                for i in range(1, len(filters_shape))]  # 将输出和权边相乘（和模型权重及其输出有关）
    # layers_out = [np.zeros(shape=(1, filters_shape[i]))
    #               for i in range(1, len(filters_shape))]  # 每层输出初始化
    # edge_num = sum([np.size(ary) for ary in edge_weights])  # 权边数
    # node_num = sum(filters_shape)  # 结点数
    #
    # for idx, edge_weight in enumerate(edge_weights):  # weights的第几层
    #     inshape = edge_weight.shape[0]
    #     oushape = edge_weight.shape[1]
    #     if idx in conv_idx:  # 卷积层
    #         edge_weights[idx] = np.linalg.norm(model_weights[weights_index[idx]], axis=(0, 1))
    #     elif idx in fc_idx:   # 全连接层
    #         edge_weights[idx] = model_weights[weights_index[idx]]

    print("end")



