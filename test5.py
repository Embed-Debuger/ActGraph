import numpy as np
from keras.models import load_model, clone_model
from Load_dataset import *
from Model.Model_Lenet5 import *
from Model.Model_vgg import *
from common import *
from Draw import *
import networkx as nx
import matplotlib.pyplot as plt
from Mutation.util import *
# from Constant import *


if __name__ == "__main__":
    from Constant import *
    """加载模型"""
    # model = lenet5_cifar10()
    # model = load_model(filepath="Model_weight/Mnist/Lenet5/weights.h5")
    # model = load_model(filepath="Model_weight/Cifar10/Lenet5/weights.h5")
    model = load_model("Model_weight/Cifar10/VGG16/weights_256.h5")

    """加载model2graph的操作类"""
    # graph_handle = lenet5_mnist_graph(model)
    # graph_handle = lenet5_cifar10_graph(model)
    graph_handle = vgg16_graph(model)

    """加载数据"""
    # 加载分类正确和错误的样本
    # _, _, _, _, X_test, Y_test = load_mnist(split=False)
    # X_test_true, Y_test_true, X_test_false, Y_test_false = filter_sample(model, X_test, Y_test)
    # # 加载对抗样本
    # _, _, _, _, X_test_adv, Y_test_adv = load_clean_adv("Adv/Mnist/Lenet5/FGSM_perclass=900.npz")

    _, _, _, _, X_test, Y_test = load_cifar10(split=False)
    X_test_true, Y_test_true, X_test_false, Y_test_false = filter_sample(model, X_test, Y_test)

    node_fea = "DC"
    threshold = 0

    # 干净样本
    sample = X_test_true[0:1]
    G, A, feature = graph_handle.Generate_graph(sample=sample, node_fea=node_fea, threshold=threshold)
    # graph_handle.Draw_graph(show=True)

    print("end")



