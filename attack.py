import numpy as np
import os
from Attack.Attack_fb import *
from Load_dataset import *
from Constant import *
from common import *


if __name__ == "__main__":
    Dataset = "Cifar100"
    Mo = "VGG19"
    operator = "CW"

    train_at_num = 30000    # 3w个训练集对抗样本
    train_tr_num = 60000-train_at_num
    test_at_num = 2000  # 2k个测试集对抗样本
    test_tr_num = 10000-test_at_num
    # 加载模型
    model = get_model(Dataset=Dataset, Mo=Mo)
    # 加载数据
    X_train, Y_train, _, _, X_test, Y_test = load_dataset(Dataset=Dataset, split=False)

    X_train_ture, Y_train_ture, X_train_false, Y_train_false = filter_sample(model, X_train, Y_train)
    X_test_ture, Y_test_ture, X_test_false, Y_test_false = filter_sample(model, X_test, Y_test)

    # 开始攻击
    att = fb_attack(model, ope=operator)
    X_train_adv, Y_train_adv = att.generate_adv(X_train_ture, Y_train_ture, train_at_num)  # 生成3W个对抗样本训练集
    X_test_adv, Y_test_adv = att.generate_adv(X_test_ture, Y_test_ture, test_at_num)  # 生成2000个对抗样本测试集

    X_train = np.concatenate((X_train_adv[:train_at_num], X_train_ture[:train_tr_num]), axis=0)
    Y_train = np.concatenate((Y_train_adv[:train_at_num], Y_train_ture[:train_tr_num]), axis=0)
    X_test = np.concatenate((X_test_adv[:test_at_num], X_test_ture[:test_tr_num]), axis=0)
    Y_test = np.concatenate((Y_test_adv[:test_at_num], Y_test_ture[:test_tr_num]), axis=0)

    # 打乱
    train_idx = np.random.choice(len(X_train), len(X_train), replace=False)
    test_idx = np.random.choice(len(X_test), len(X_test), replace=False)

    X_train = X_train[train_idx]
    Y_train = Y_train[train_idx]
    X_test = X_test[test_idx]
    Y_test = Y_test[test_idx]

    # 保存
    np.savez("Adv/{}/{}/{}_{}.npz".format(Dataset, Mo, Dataset, operator),
             X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print("end")



