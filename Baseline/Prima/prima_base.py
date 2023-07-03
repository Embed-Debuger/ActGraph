import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径
from keras.models import Model, Input
from GradPri_utils.utils import *
from tqdm import tqdm
import xgboost
from common import *


def prima_model_rank(base_model, trainset, testset, ideal_rank, Dataset="Mnist", Mo="lenet5"):
    # 基于模型变异的测试方法
    if Dataset=="Mnist":
        classes = 10
        if Mo=="Lenet5":
            layer_name = 'dense_1'
            layer_num = 8
            clon_model = clone_model(base_model)  # 克隆模型
            clon_model.set_weights(base_model.get_weights())
            hidden_layer_Model, partial_model_Model = Split_model(base_model, layer_num=layer_num)
    elif Dataset=="Cifar10":
        classes = 10
        if Mo == "VGG16":
            layer_name = 'fc2'
            layer_num = 40
            clon_model = clone_model(base_model)  # 克隆模型
            clon_model.set_weights(base_model.get_weights())
            hidden_layer_Model, partial_model_Model = Split_model(base_model, layer_num=layer_num)
        elif Mo == "Resnet18":
            layer_name = 'fc1'
            layer_num = 65
            clon_model = clone_model(base_model)  # 克隆模型
            clon_model.set_weights(base_model.get_weights())
            hidden_layer_Model, partial_model_Model = Split_model(base_model, layer_num=layer_num)
    elif Dataset=="Cifar100":
        classes = 100
        if Mo == "VGG19":
            layer_name = 'fc2'
            layer_num = 46
            clon_model = clone_model(base_model)  # 克隆模型
            clon_model.set_weights(base_model.get_weights())
            hidden_layer_Model, partial_model_Model = Split_model(base_model, layer_num=layer_num)

    # 数据集
    train_len = len(trainset[0])
    X_train = trainset[0]   # 图片
    Y_train = trainset[1]   # 类标
    bug_train = trainset[2] # bug类标
    X_test = testset[0]
    Y_test = testset[1]
    bug_test = testset[2]
    X = np.concatenate((X_train, X_test), axis=0)

    model_feature = get_model_feature(base_model, clon_model, hidden_layer_Model, partial_model_Model, X, layer_name,
                                      classes)
    sample_feature = get_sample_feature(base_model, X)

    feature = np.concatenate((model_feature, sample_feature), axis=1)

    train_feature = feature[:train_len]
    test_feature = feature[train_len:]

    # 训练二分类器
    xg_reg = xgboost.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.5, learning_rate=0.05,
                                  max_depth=5, alpha=1, n_estimators=10)
    xg_reg.fit(train_feature, bug_train)

    # 排序
    pred = xg_reg.predict(test_feature) # 预测样本
    select_ls = np.argsort(-pred)   # 从大到小排列的索引
    x = X_test[select_ls]
    y = Y_test[select_ls]
    bug = bug_test[select_ls]

    # 计算指标
    real_rank = bug_test[select_ls]    # 测试结果的错误排列顺序
    apfd_all = RAUC(ideal_rank, real_rank)
    print("APFD分数：%.3f%%" % (apfd_all*100))

    return select_ls, (x, y, bug), (train_feature, Y_train, bug_train)


def get_model_feature(base_model, clon_model, hidden_layer_Model, partial_model_Model, X, layer_name, classes=10):
    model_mutant_num = 100  # 单个模型得到的变异体数量，PRIMA中设置变异模型为100个
    weight_mutant_rate = 0.1  # 模型中的权重修改比例，，PRIMA中设置为10%
    model_feature = []
    for i in tqdm(range(0, len(X))):
        x_tmp = np.array([X[i]])
        pre_tmp = base_model.predict(x_tmp)
        a = pre_tmp  # ori_prob[i] # 样本的预测置信度
        max_value = np.max(a)  # 预测置信度的最大概率值
        max_value_pos = np.argmax(a)  # 样本的预测标签

        # 进行模型的变异操作
        perturbated_predictions = np.zeros((4, int(model_mutant_num/4), classes))
        for ii in range(0, int(model_mutant_num/4)): # 有四种模型变异操作
            conf_NAI = NeuActInverse_confidence(x_tmp, hidden_layer_Model,
                                                partial_model_Model, weight_mutant_rate)  # 神经元激活翻转
            conf_NEB = NeuEffBlock_confidence(x_tmp, hidden_layer_Model,
                                              partial_model_Model, weight_mutant_rate)  # 神经元激活置零
            conf_GF = GaussFuzz_confidence(x_tmp, base_model, clon_model, 0, 0.1, layer_name)  # 神经元激活添加高斯噪声
            conf_WS = WeightShuffl_confidence(x_tmp, base_model, clon_model, layer_name, weight_mutant_rate)  # 部分权重值打乱
            perturbated_predictions[0][ii] = conf_NAI[0]
            perturbated_predictions[1][ii] = conf_NEB[0]
            perturbated_predictions[2][ii] = conf_GF[0]
            perturbated_predictions[3][ii] = conf_WS[0]

        # 对于待测样本的变异扰动样本的特征提取
        feature_tmp = []
        for perturbated_prediction in perturbated_predictions:  # 多种变异策略
            euler = 0
            mahat = 0
            qube = 0
            cos = 0
            difference = 0
            different_class = []
            cos_list = []
            for pp in perturbated_prediction:  # 取每一种变异扰动样本的置信度值
                pro = pp
                opro = a

                difference += abs(max_value - pp[max_value_pos])
                euler += np.linalg.norm(pro - opro)
                mahat += np.linalg.norm(pro - opro, ord=1)
                qube += np.linalg.norm(pro - opro, ord=np.inf)
                co = np.clip(1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))), 0, 1)
                cos += co
                cos_list.append(co)
                if np.argmax(pp) != max_value_pos:
                    different_class.append(np.argmax(pp))  # 变异后的样本错分为其他类
            cos_dis = cos_distribution(cos_list)
            dic = {}
            for key in different_class:
                dic[key] = dic.get(key, 0) + 1  # 统计被错分成哪几类样本
            wrong_class_num = len(dic)  # 错分为wrong_class_num个类
            if len(dic) > 0:
                max_class_num = max(dic.values())  # 统计错分到哪类的数量最大
            else:
                max_class_num = 0

            feature_tmp.extend([euler])
            feature_tmp.extend(i for i in [mahat, qube, cos[0], difference, wrong_class_num, max_class_num])
            feature_tmp.extend(cos_dis)
        model_feature.append(feature_tmp)
    model_feature = np.array(model_feature)
    return model_feature


def get_sample_feature(base_model, X):
    img_size = np.shape(X)[1:3]     # 图像尺寸

    sample_mutant_num = 200  # 单个样本得到的变异体数量，PRIMA中设置变异样本为200个
    sample_feature = []

    for i in tqdm(range(0, len(X))):
        x_tmp = np.array([X[i]])
        pre_tmp = base_model.predict(x_tmp)
        a = pre_tmp  # ori_prob[i] # 样本的预测置信度
        max_value = np.max(a)  # 预测置信度的最大概率值
        max_value_pos = np.argmax(a)  # 样本的预测标签

        perturbated_img0 = []
        perturbated_img1 = []
        perturbated_img2 = []
        perturbated_img3 = []
        perturbated_img4 = []
        for mutants_input in range(0, int(sample_mutant_num / 5)):  # 循环sample_mutant_num次，生成sample_mutant_num个变异样本
            row1 = np.random.randint(0, img_size[0], dtype=int)  # 确定扰动的像素纵坐标（行）
            col1 = np.random.randint(0, img_size[1], dtype=int)  # 确定扰动的像素横坐标（列）

            perturbated_img0.append(gauss_noise(x_tmp, row1, col1, ratio=1.0, var=0.01))
            perturbated_img1.append(white(x_tmp, row1, col1))
            perturbated_img2.append(black(x_tmp, row1, col1))
            perturbated_img3.append(reverse_color(x_tmp, row1, col1))
            perturbated_img4.append(shuffle_pixel(x_tmp, row1, col1))
        perturbated_img = [perturbated_img0, perturbated_img1, perturbated_img2, perturbated_img3, perturbated_img4]
        perturbated_img = np.squeeze(np.array(perturbated_img), axis=2)
        perturbated_predictions = np.array([base_model.predict(x) for x in perturbated_img])  # 样本扰动后的预测置信度 (5, 40, 10)

        feature_tmp = []
        for perturbated_prediction in perturbated_predictions:
            # 对于待测样本的变异扰动样本的特征提取
            euler = 0
            mahat = 0
            qube = 0
            cos = 0
            difference = 0
            different_class = []
            cos_list = []
            for pp in perturbated_prediction:  # 取每一种变异扰动样本的置信度值
                pro = pp
                opro = a
                difference += abs(max_value - pp[max_value_pos])
                euler += np.linalg.norm(pro - opro)  # 正常样本对变异样本的预测概率范数
                mahat += np.linalg.norm(pro - opro, ord=1)
                qube += np.linalg.norm(pro - opro, ord=np.inf)
                co = np.clip(1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))), 0, 1)
                cos += co
                cos_list.append(co)
                if np.argmax(pp) != max_value_pos:  # 如果变异后结果不一样
                    different_class.append(np.argmax(pp))  # 保存变异类的预测类标
            cos_dis = cos_distribution(cos_list)
            dic = {}
            for key in different_class:
                dic[key] = dic.get(key, 0) + 1
            wrong_class_num = len(dic)
            if len(dic) > 0:
                max_class_num = max(dic.values())
            else:
                max_class_num = 0

            feature_tmp.extend([euler])
            feature_tmp.extend(i for i in [mahat, qube, cos[0], difference, wrong_class_num, max_class_num])
            feature_tmp.extend(cos_dis)
        sample_feature.append(feature_tmp)
    sample_feature = np.array(sample_feature)
    return sample_feature


if __name__ == "__main__":
    print("end")



