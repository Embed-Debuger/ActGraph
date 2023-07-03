import numpy as np
from keras.datasets import cifar10, mnist, cifar100
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from  common import *


def load_dataset(Dataset, onehot=False, sequence=False, split=True):
    X_train = None
    Y_train = None
    X_val   = None
    Y_val   = None
    X_test  = None
    Y_test  = None

    if Dataset=="Cifar10":
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = np.reshape(X_train, (-1, 32, 32, 3))
        X_test = np.reshape(X_test, (-1, 32, 32, 3))
        Y_train = np.reshape(Y_train, (-1,))
        Y_test = np.reshape(Y_test, (-1,))
    elif Dataset == "Mnist":
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = np.reshape(X_train, (-1, 28, 28, 1))
        X_test = np.reshape(X_test, (-1, 28, 28, 1))
        Y_train = np.reshape(Y_train, (-1,))
        Y_test = np.reshape(Y_test, (-1,))
    elif Dataset == "Cifar100":
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
        X_train = np.reshape(X_train, (-1, 32, 32, 3))
        X_test = np.reshape(X_test, (-1, 32, 32, 3))
        Y_train = np.reshape(Y_train, (-1,))
        Y_test = np.reshape(Y_test, (-1,))


    if sequence == True:
        # 按类别顺序排列
        X_train = X_train[np.argsort(Y_train)]
        Y_train = Y_train[np.argsort(Y_train)]
        X_test = X_test[np.argsort(Y_test)]
        Y_test = Y_test[np.argsort(Y_test)]
    else:
        # 打乱
        tra_idx = np.random.choice(len(Y_train), size=len(Y_train), replace=False)
        tes_idx = np.random.choice(len(Y_test), size=len(Y_test), replace=False)
        X_train = X_train[tra_idx]
        Y_train = Y_train[tra_idx]
        X_test = X_test[tes_idx]
        Y_test = Y_test[tes_idx]

    if onehot == True:
        # 类标转为onehot矩阵
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)

    # 数据归一化
    X_train = normalization(X_train)
    X_test = normalization(X_test)

    if split == True:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)  # 将训练集切割为训练集和验证集

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def filter_sample(model, X, Y, sequence=False):
    # 筛选模型能正确识别的样本
    if sequence == True:
        X = X[np.argsort(Y)]  # 排序
        Y = Y[np.argsort(Y)]

    pre = model.predict(X)
    pre_label = np.argmax(pre, axis=1)

    X_ture = X[np.where(pre_label == Y)]
    Y_ture = Y[np.where(pre_label == Y)]
    X_false = X[np.where(pre_label != Y)]
    Y_false = Y[np.where(pre_label != Y)]
    return X_ture, Y_ture, X_false, Y_false


def filter_low_loss(model, sample, label, rate=0):
    # 过滤低loss的样本
    loss = np.array([model.evaluate(x=np.array([x]), y=np.array([y]), verbose=0)[0] for x, y in zip(sample, label)])
    thres = loss[np.argsort(-loss)[int(len(sample)*rate)]]    # 阈值
    # print(thres)
    idx = np.where(loss < thres)
    return sample[idx], label[idx]


if __name__ == "__main__":
    # from Lenet5 import *
    # model = lenet5_cifar10(path="Model/Cifar10/Lenet5/Wed-Feb-23-20:33:29-2022.h5")
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10()

    # X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(Dataset="Cifar100")
    print("end")


