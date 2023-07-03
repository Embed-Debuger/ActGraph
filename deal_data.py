import numpy as np
from keras.models import save_model
from Load_dataset import *
from common import *
from Constant import *


def deal_mnist():
    model = get_model(Dataset="Mnist", Mo="lenet5")
    X_train, Y_train, _, _, X_test, Y_test = load_mnist(split=False)
    X_train_true, Y_train_true, X_train_false, Y_train_false = filter_sample(model, X_train, Y_train)    # 找到所有假阳性样本

    X_train_false = np.array([X_train_false]*1000).reshape((-1,28,28,1))[:30000]
    Y_train_false = np.array([Y_train_false]*1000).reshape((-1,))[:30000]
    X_train_true = np.array([X_train_true]*10).reshape((-1,28,28,1))[:30000]
    Y_train_true = np.array([Y_train_true]*10).reshape((-1,))[:30000]
    X_train = np.concatenate((X_train_true, X_train_false), axis=0)
    Y_train = np.concatenate((Y_train_true, Y_train_false), axis=0)

    idx = np.random.choice(len(X_train), size=len(X_train), replace=False)
    X_train = X_train[idx]
    Y_train = Y_train[idx]

    np.savez("Adv/Mnist/Lenet5/Mnist_original.npz", X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)


if __name__ == "__main__":
    Dataset = "Cifar100"
    Mo = "VGG19"
    Operator = "origin"
    model = get_model(Dataset=Dataset, Mo=Mo)
    X_train, Y_train, _, _, X_test, Y_test = load_dataset(Dataset=Dataset, onehot=False, split=False)

    X_train_true, Y_train_true, X_train_false, Y_train_false = filter_sample(model, X_train, Y_train)  # 找到所有假阳性样本
    # X_test_true, Y_test_true, X_test_false, Y_test_false = filter_sample(model, X_test, Y_test)

    X_train_false = np.array([X_train_false]*100).reshape((-1,32,32,3))[:30000]
    Y_train_false = np.array([Y_train_false]*100).reshape((-1,))[:30000]
    X_train_true = np.array([X_train_true]*10).reshape((-1,32,32,3))[:30000]
    Y_train_true = np.array([Y_train_true]*10).reshape((-1,))[:30000]
    X_train = np.concatenate((X_train_true, X_train_false), axis=0)
    Y_train = np.concatenate((Y_train_true, Y_train_false), axis=0)

    idx = np.random.choice(len(X_train), size=len(X_train), replace=False)
    X_train = X_train[idx]
    Y_train = Y_train[idx]

    np.savez("Adv/{}/{}/{}_{}.npz".format(Dataset, Mo, Dataset, Operator),
             X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    # np.savez("Adv/Cifar10/Resnet18/Cifar10_origin.npz", X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    # save_model(model, filepath="/home/NewDisk/gejie/program/Graph_mutation/Model_weight/Cifar10/VGG19/weights.h5")
    print("end")






