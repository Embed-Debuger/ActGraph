import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from sa import fetch_dsa
import numpy as np
import argparse
from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--d", "-d", help="Dataset", type=str, default="Mnist")
parser.add_argument(
    "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
)
parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm"
)
parser.add_argument(
    "--save_path", "-save_path", help="Save path", type=str, default="/home/NewDisk/gejie/program/Graph_mutation/Baseline/DSA/tmp/"
)
parser.add_argument(
    "--batch_size", "-batch_size", help="Batch size", type=int, default=128
)
parser.add_argument(
    "--var_threshold",
    "-var_threshold",
    help="Variance threshold",
    type=int,
    default=1e-5,
)
parser.add_argument(
    "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
)
parser.add_argument(
    "--n_bucket",
    "-n_bucket",
    help="The number of buckets for coverage",
    type=int,
    default=1000,
)
parser.add_argument(
    "--num_classes",
    "-num_classes",
    help="The number of classes",
    type=int,
    default=10,
)
parser.add_argument(
    "--is_classification",
    "-is_classification",
    help="Is classification task",
    type=bool,
    default=True,
)
args = parser.parse_args()

def dsa_rank(model, trainset, testset, ideal_rank, Dataset="Mnist", Mo="Lenet5", Operator="origin"):
    if Dataset=="Mnist":
        args.d = Dataset
        args.num_classes = 10
        args.target = Operator
        if Mo=="Lenet5":
            layer_names = ["dense_1"]
    elif Dataset=="Cifar10":
        args.d = Dataset
        args.num_classes = 10
        args.target = Operator
        if Mo=="VGG16":
            layer_names = ["fc2-relu"]
        elif Mo=="Resnet18":
            layer_names = ["flatten_1"]
    elif Dataset=="Cifar100":
        args.d = Dataset
        args.num_classes = 100
        args.target = Operator
        if Mo=="VGG19":
            layer_names = ["fc2-relu"]

    train_len = len(trainset[0])  # 训练集数量
    X_train = trainset[0]  # 图像
    Y_train = trainset[1]  # 类标
    bug_train = trainset[2]  # bug标记

    X_test = testset[0]
    Y_test = testset[1]
    bug_test = testset[2]

    test_dsa = fetch_dsa(model, X_train, Y_train, X_test, "test", layer_names, args)    # dsa越大，惊喜越高
    select_ls = np.argsort(-test_dsa)   # 由大到小排列

    x = X_test[select_ls]
    y = Y_test[select_ls]
    bug = bug_test[select_ls]

    # 计算指标
    apfd_all = RAUC(ideal_rank, bug)
    print("APFD分数：%.3f%%" % (apfd_all * 100))

    return select_ls, (x, y, bug)


if __name__ == "__main__":

    print("end")





