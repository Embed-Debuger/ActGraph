import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径

import numpy as np
from common import *


def Gini_score(confidence):
    confidence = np.array(confidence)
    gini = np.sum(confidence * confidence, axis=1)
    return 1-gini


def gini_rank(model, testset, ideal_rank):
    X_test = testset[0]  # 图像
    Y_test = testset[1]  # 类标
    bug_test = testset[2]  # bug类标

    confidence = model.predict(X_test, batch_size=128)
    gini_score = Gini_score(confidence)
    select_ls = np.argsort(-gini_score)     # 由大到小排

    x = X_test[select_ls]
    y = Y_test[select_ls]
    bug = bug_test[select_ls]
    # 计算指标
    real_rank = bug_test[select_ls]  # 测试结果的错误排列顺序
    apfd_all = RAUC(ideal_rank, real_rank)
    print("RAUC_ALL：%.3f%%" % (apfd_all * 100))

    return select_ls, (x, y, bug)


if __name__ == "__main__":
    print("gini")



