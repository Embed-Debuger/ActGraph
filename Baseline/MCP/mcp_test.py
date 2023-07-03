import numpy as np

from mcp_base import *
from Model.Model_Lenet5 import *
from Load_dataset import *
from common import *
from Draw import *
from tqdm import tqdm

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    # 加载模型
    # model_path = os.path.join(base_path, "../../Model_weight/Cifar10/Lenet5/Wed-Feb-23-20:33:29-2022.h5")
    model_path = os.path.join(base_path, "../../Model_weight/Mnist/Lenet5/weights.h5")
    # model = lenet5_cifar10(path=model_path)
    model = lenet5_mnist(path=model_path)

    # 加载数据
    # data_path = os.path.join(base_path, "../../Adv/Cifar10/Lenet5/PGD_perclass=400.npz")
    data_path = os.path.join(base_path, "../../Adv/Mnist/Lenet5/FGSM_perclass=900.npz")
    X_train_clean, X_train_adv, Class_train, X_test_clean, X_test_adv, Class_test = load_clean_adv(data_path)

    # 测试
    test_adv_num = 1000
    test_clean_num = 4000

    X_clean_test = X_test_clean[:test_clean_num]
    X_adv_test = X_test_adv[:test_adv_num]
    X_test = np.concatenate((X_adv_test, X_clean_test), axis=0)
    Class_test = np.concatenate((Class_test[:test_adv_num], Class_test[:test_clean_num]), axis=0)
    Y_test = np.array([1]*test_adv_num + [0]*test_clean_num)

    # 排序
    selecnum = len(X_test)
    _, _, rank = select_mnist(model, selecnum, X_test, Class_test)

    ideal = Y_test[:selecnum]
    real = Y_test[rank]

    apfd = improved_APFD(ideal, real)

    print("APFD分数：%.5f" % apfd)
    plot_data(path="mcp.png", data1=curve(ideal), label1="Ideal",
              data2=curve(real), label2="Gini",
              Xlabel="Sample", Ylabel="The cumulative number of errors", show=False)

    print("end")




