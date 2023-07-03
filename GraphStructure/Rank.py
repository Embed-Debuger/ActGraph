import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径
from Model.Model_Lenet5 import *
from Model.Model_vgg import *
from Model.Model_Resnet import *
import xgboost
from xgboost import plot_importance
from sklearn import svm
from common import *


def graph_rank(model, trainset, testset, ideal_rank, Dataset="Mnist", Mo="lenet5"):
    if Dataset=="Mnist":
        if Mo=="Lenet5":
            graph_handle = lenet5_mnist_graph(model)
    elif Dataset=="Cifar10":
        if Mo=="Lenet5":
            graph_handle = lenet5_cifar10_graph(model)
        elif Mo == "VGG16":
            graph_handle = vgg16_graph(model, Dataset=Dataset)
        elif Mo=="Resnet18":
            graph_handle = resnet18_cifar10_graph(model)
    elif Dataset == "Cifar100":
        if Mo=="VGG19":
            graph_handle = vgg19_cifar100_graph(model)

    train_len = len(trainset[0])    # 训练集数量
    X_train = trainset[0]   # 图像
    Y_train = trainset[1]   # 类标
    bug_train = trainset[2] # bug标记

    X_test = testset[0]
    Y_test = testset[1]
    bug_test = testset[2]
    X = np.concatenate((X_train, X_test), axis=0)

    # 提取图节点特征
    node_fea = "DC"
    threshold = -10
    feature = np.array([graph_handle.Generate_graph(np.array([x]), node_fea=node_fea, threshold=threshold)[2]
                        for x in tqdm(X)])
    # feature = batch_standardization(feature)
    train_feature = feature[:train_len]
    test_feature = feature[train_len:]

    # 训练排序器
    xg_reg = xgboost.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=1, n_estimators=10)
    xg_reg.fit(train_feature, bug_train)

    # # plot
    # from matplotlib import pyplot
    # # print(xg_reg.feature_importances_)
    # # pyplot.bar(range(len(xg_reg.feature_importances_)), xg_reg.feature_importances_)
    # # pyplot.show()
    # print(xg_reg.feature_importances_)
    # plot_importance(xg_reg)
    # pyplot.show()

    # 排序
    pred = xg_reg.predict(test_feature) # 预测样本
    select_ls = np.argsort(-pred)   # 从大到小排列的索引
    x = X_test[select_ls]
    y = Y_test[select_ls]
    bug = bug_test[select_ls]

    # # svm
    # classifiers = svm.SVC(kernel='linear', C=0.5, gamma=0.5, probability=True)
    # classifiers.fit(train_feature, bug_train)
    # print("训练集精度：%.5f" % classifiers.score(train_feature, bug_train))
    # print("测试集精度：%.5f" % classifiers.score(test_feature, bug_test))
    # # 排序
    # pred = classifiers.predict_proba(test_feature)
    # select_ls = np.argsort(-pred[..., 1:].reshape(-1, ))  # 对 对抗类的预测概率 由大到小排列
    # x = X_test[select_ls]
    # y = Y_test[select_ls]
    # bug = bug_test[select_ls]

    # 计算指标
    apfd_all = RAUC(ideal_rank, bug)
    print("APFD分数：%.3f%%" % (apfd_all*100))

    return select_ls, (x, y, bug), (train_feature, Y_train, bug_train)


def act_rank(model, trainset, testset, ideal_rank, Dataset="Mnist", Mo="lenet5"):
    if Dataset=="Mnist":
        if Mo=="Lenet5":
            graph_handle = lenet5_mnist_graph(model)
    elif Dataset=="Cifar10":
        # if Mo=="lenet5":
        #     graph_handle = lenet5_cifar10_graph(model)
        if Mo == "VGG16":
            graph_handle = vgg16_graph(model, Dataset=Dataset)
        elif Mo=="Resnet18":
            graph_handle = resnet18_cifar10_graph(model)
    elif Dataset=="Cifar100":
        if Mo == "VGG19":
            graph_handle = vgg19_cifar100_graph(model)

    train_len = len(trainset[0])    # 训练集数量
    X_train = trainset[0]   # 图像
    Y_train = trainset[1]   # 类标
    bug_train = trainset[2] # bug标记

    X_test = testset[0]
    Y_test = testset[1]
    bug_test = testset[2]
    X = np.concatenate((X_train, X_test), axis=0)

    # 提取神经元激活特征
    feature = np.array([graph_handle.Generate_act(np.array([x])) for x in tqdm(X)])
    train_feature = feature[:train_len]
    test_feature = feature[train_len:]

    # 训练排序器
    xg_reg = xgboost.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=1, n_estimators=10)
    xg_reg.fit(train_feature, bug_train)

    # 排序
    pred = xg_reg.predict(test_feature) # 预测样本
    select_ls = np.argsort(-pred)   # 从大到小排列的索引
    x = X_test[select_ls]
    y = Y_test[select_ls]
    bug = bug_test[select_ls]

    # 计算指标
    apfd_all = RAUC(ideal_rank, bug)
    print("APFD分数：%.3f%%" % (apfd_all*100))

    return select_ls, (x, y, bug), (train_feature, Y_train, bug_train)


if __name__ == "__main__":
    print("end")







