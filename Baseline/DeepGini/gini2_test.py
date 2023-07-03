from Load_dataset import *
from Model.Model_Lenet5 import *
from common import *
import random
import numpy as np
from tqdm import tqdm
from sklearn import svm
from Draw import *


if __name__ == "__main__":
    # 加载模型
    model = lenet5_cifar10(path="Model_weight/Cifar10/Lenet5/Wed-Feb-23-20:33:29-2022.h5")
    # 加载数据
    X_train_clean, X_train_adv, Class_train, X_test_clean, X_test_adv, Class_test = load_clean_adv(
        "Adv/Cifar10/Lenet5/CW_perclass=400.npz")
    _, _, _, _, X_fp, Y_fp = load_cifar10(onehot=False, sequence=False, split=False)
    _, _, X_fp, Y_fp = filter_sample(model, X_fp, Y_fp)

    test_adv_num = 1000
    test_fp_num = 1000
    test_clean_num = 4000

    X_clean_test = X_test_clean[:test_clean_num]
    X_fp_test = X_fp[:test_fp_num]
    X_adv_test = X_test_adv[:test_adv_num]
    X_test = np.concatenate((X_adv_test, X_fp_test, X_clean_test), axis=0)
    Y_test = np.array([1]*test_adv_num + [1]*test_fp_num + [0]*test_clean_num)

    confidence = model.predict(X_test, batch_size=128)
    gini_score = Gini_score(confidence)

    rank = np.argsort(-gini_score)
    apfd = improved_APFD(Y_test, Y_test[rank])
    print("APFD分数：%.3f%%" % (apfd*100))

    # 绘制曲线
    plot_data(path="图/gini.png", data1=curve(Y_test), label1="Ideal",
              data2=curve(Y_test[rank]), label2="Gini",
              Xlabel="Sample", Ylabel="The cumulative number of errors", show=False)
    print("end")



