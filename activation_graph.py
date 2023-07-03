import matplotlib.pyplot as plt

from Load_dataset import *
from Model.Model_Lenet5 import *
from Attack.Attack_fb import *
from Constant import *

if __name__ == "__main__":
    # 观察各个样本的激活图
    Dataset = "Mnist"
    Mo = "Lenet5"
    Operator = "origin"
    idx = 14
    # 加载数据
    # X_train, Y_train, _, _ = get_dataset(Dataset, Mo, Operator)
    _, _, X_test, Y_test = get_dataset(Dataset, Mo, Operator)
    sample = X_test[idx:idx+1]
    label = Y_test[idx:idx+1]

    # 加载模型
    model = get_model(Dataset, Mo)

    # 加载graph类
    graph_handle = lenet5_mnist_graph(model)

    # 对抗样本
    # att = fb_attack(model, ope="CW")
    # sample, _ = att.generate_adv(sample, label, 1)

    # plt.imshow(sample[0], cmap="gray")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    graph_handle.Generate_graph(sample, node_fea="DC", threshold=0.4)
    graph_handle.Draw_graph(show=True)
    print("end")


