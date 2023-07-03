import numpy as np
from tqdm import tqdm
from keras.models import load_model


def standardization(data):
    # 标准化
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def batch_standardization(datas):
    # datas是两维数据
    # 对每个特征进行标准化
    bat_num = datas.shape[0]
    fea_num = datas.shape[1]
    for i in range(fea_num):
        datas[..., i] = standardization(datas[..., i])
    return datas


def normalization(data):
    # 归一化
    _range = np.max(data) - np.min(data)
    return (np.array(data) - np.min(data)) / (_range+1e-15)


def APFD(data):
    data = np.array(data)
    num = np.size(data)     # 元素数
    adv_num = np.sum(data != 0)     # 错误样本数
    idx = np.where(data != 0)
    # print(idx)
    score = 1 - ((np.sum(idx))/(adv_num*num)) + (1/(2*num))
    return score


def RAUC(ideal_data, real_data):
    ideal_curve = curve(ideal_data)
    real_curve = curve(real_data)
    ideal_area = curve_area(ideal_curve)
    real_area = curve_area(real_curve)
    return real_area/ideal_area


def curve(data):
    data = np.array(data)
    for i in range(len(data)):
        idx = len(data) - i - 1
        data[idx] = np.sum(data[:idx + 1])
    return data


def curve_area(data):
    # 计算曲线下面积
    data = curve(data)
    return np.sum(data)


def graph_select(classifier, graph_handle, selectsize, X, Y, node_fea="DC", thres=0.2):
    X_test_sig = np.array([graph_handle.Generate_graph(np.array([x]), node_fea=node_fea, threshold=thres)[2]
                           for x in tqdm(X)])
    pred = classifier.predict_proba(X_test_sig)
    pred_adv = pred[..., 1:].reshape(-1,)   # 对 对抗类的预测概率
    rank = np.argsort(-pred_adv)     # 排序后的索引
    select_ls = rank[:selectsize]
    return X[:selectsize], Y[:selectsize], select_ls


def get_model(Dataset, Mo):
    if Dataset=="Mnist":
        if Mo=="Lenet5":
            model = load_model("../Model_weight/Mnist/Lenet5/weights.h5")
    elif Dataset=="Cifar10":
        if Mo=="VGG16":
            model = load_model("Model_weight/Cifar10/VGG16/weights_256.h5")
        elif Mo=="Resnet18":
            model = load_model("Model_weight/Cifar10/Resnet18/weights.h5")
    elif Dataset == "Cifar100":
        if Mo == "VGG19":
            model = load_model("Model_weight/Cifar100/VGG19/weights.h5")
    return model


def get_dataset(Dataset, Mo, Operator):
    # base_path = "../"
    base_path = ""
    if Dataset=="Mnist":
        if Mo == "Lenet5":
            if Operator == "origin":
                data = np.load(base_path + "../Adv/Mnist/Lenet5/Mnist_origin.npz", allow_pickle=True)
            elif Operator == "Rotate":
                data = np.load(base_path + "../Adv/Mnist/Lenet5/Mnist_Rotate.npz", allow_pickle=True)
            elif Operator == "JSMA":
                data = np.load(base_path + "../Adv/Mnist/Lenet5/Mnist_JSMA.npz", allow_pickle=True)
            elif Operator == "CW":
                data = np.load(base_path + "../Adv/Mnist/Lenet5/Mnist_CW.npz", allow_pickle=True)
            elif Operator == "Mix-JCR":
                data = np.load(base_path + "../Mnist/Lenet5/Mnist_Mix-JCR.npz", allow_pickle=True)
    elif Dataset=="Cifar10":
        if Mo=="VGG16":
            if Operator == "origin":
                data = np.load(base_path + "Adv/Cifar10/VGG16/Cifar10_origin.npz", allow_pickle=True)
            elif Operator == "Rotate":
                data = np.load(base_path + "Adv/Cifar10/VGG16/Cifar10_Rotate.npz", allow_pickle=True)
            elif Operator == "JSMA":
                data = np.load(base_path + "Adv/Cifar10/VGG16/Cifar10_JSMA.npz", allow_pickle=True)
            elif Operator == "CW":
                data = np.load(base_path + "Adv/Cifar10/VGG16/Cifar10_CW.npz", allow_pickle=True)
            elif Operator == "Mix-JCR":
                data = np.load(base_path + "Adv/Cifar10/VGG16/Cifar10_Mix-JCR.npz", allow_pickle=True)
        elif Mo=="Resnet18":
            if Operator == "origin":
                data = np.load(base_path + "Adv/Cifar10/Resnet18/Cifar10_origin.npz", allow_pickle=True)
            elif Operator == "Rotate":
                data = np.load(base_path + "Adv/Cifar10/Resnet18/Cifar10_Rotate.npz", allow_pickle=True)
            elif Operator == "JSMA":
                data = np.load(base_path + "Adv/Cifar10/Resnet18/Cifar10_JSMA.npz", allow_pickle=True)
            elif Operator == "CW":
                data = np.load(base_path + "Adv/Cifar10/Resnet18/Cifar10_CW.npz", allow_pickle=True)
            elif Operator == "Mix-JCR":
                data = np.load(base_path + "Adv/Cifar10/Resnet18/Cifar10_Mix-JCR.npz", allow_pickle=True)
    elif Dataset=="Cifar100":
        if Mo=="VGG19":
            if Operator == "origin":
                data = np.load(base_path + "Adv/Cifar100/VGG19/Cifar100_origin.npz", allow_pickle=True)
            elif Operator == "Rotate":
                data = np.load(base_path + "Adv/Cifar100/VGG19/Cifar100_Rotate.npz", allow_pickle=True)
            elif Operator == "JSMA":
                data = np.load(base_path + "Adv/Cifar100/VGG19/Cifar100_JSMA.npz", allow_pickle=True)
            elif Operator == "CW":
                data = np.load(base_path + "Adv/Cifar100/VGG19/Cifar100_CW.npz", allow_pickle=True)
            elif Operator == "Mix-JCR":
                data = np.load(base_path + "Cifar100/VGG19/Cifar100_Mix-JCR.npz", allow_pickle=True)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    print("end")





