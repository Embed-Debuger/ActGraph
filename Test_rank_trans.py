import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径

from Baseline.Prima.GradPri_utils.utils import *
from Baseline.Prima.prima_base import *
from Baseline.DeepGini.gini_base import *
from Baseline.MCP.mcp_base import *
from Baseline.SRS.srs_base import *
from Baseline.Neuro_Frequent.neuro_base import *
from Baseline.DSA.dsa_base import *

from keras.models import load_model
from Load_dataset import *
from GraphStructure.Rank import *
from Mutation.Rank import *
from Draw import *
from Constant import *
import argparse
from time import time


if __name__ == "__main__":
    Dataset = "Cifar10"
    Mo = "Resnet18"
    Operator_train = "Mix-JCR"
    Operator_test = "Mix-JCR"
    train_num = 500
    test_num = 10000

    print(Dataset, Mo, [Operator_train, Operator_test], "train_num:", train_num, "test_num:", test_num)
    model = get_model(Dataset, Mo)
    X_train, Y_train, _, _ = get_dataset(Dataset, Mo, Operator_train)
    _, _, X_test, Y_test = get_dataset(Dataset, Mo, Operator_test)

    # print(model.evaluate(X_train, Y_train))
    # print(model.evaluate(X_test, Y_test))

    bug_train = np.where(np.argmax(model.predict(X_train), axis=1) == Y_train, 0, 1)  # 样本是否触发模型漏洞
    bug_test = np.where(np.argmax(model.predict(X_test), axis=1) == Y_test, 0, 1)  # 样本是否触发模型漏洞

    trainset = (X_train[:train_num], Y_train[:train_num], bug_train[:train_num])
    testset = (X_test[:test_num], Y_test[:test_num], bug_test[:test_num])   # 样本 类标 bug标签
    ideal_rank = np.sort(bug_test[:test_num])[::-1]    # 测试用例的理想曲线

    start = time()
    # print("DeepGini")
    # result = gini_rank(model, testset, ideal_rank)
    # print("RAUC_100:", RAUC(ideal_rank[:100], result[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], result[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], result[1][2][:1000]))

    # # print("SRS")
    # # srs = srs_rank(testset, ideal_rank)
    # # print("RAUC_100:", RAUC(ideal_rank[:100], srs[1][2][:100]))
    # # print("RAUC_500:", RAUC(ideal_rank[:500], srs[1][2][:500]))
    # # print("RAUC_1000:", RAUC(ideal_rank[:1000], srs[1][2][:1000]))
    #
    # print("激活")
    # result = act_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo)
    # print("RAUC_100:", RAUC(ideal_rank[:100], result[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], result[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], result[1][2][:1000]))

    # print("MCP")
    # result = mcp_rank(model, testset, selectsize=len(testset[0]), ideal_rank=ideal_rank)
    # print("RAUC_100:", RAUC(ideal_rank[:100], result[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], result[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], result[1][2][:1000]))

    print("Prima")
    result = prima_model_rank(model, trainset, testset, ideal_rank=ideal_rank, Dataset=Dataset, Mo=Mo)
    print("RAUC_100:", RAUC(ideal_rank[:100], result[1][2][:100]))
    print("RAUC_500:", RAUC(ideal_rank[:500], result[1][2][:500]))
    print("RAUC_1000:", RAUC(ideal_rank[:1000], result[1][2][:1000]))

    # print("Graph")
    # result = graph_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo)
    # print("RAUC_100:", RAUC(ideal_rank[:100], result[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], result[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], result[1][2][:1000]))

    # print("Muta_Graph")
    # muta_graph = mutation_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo)
    # print("RAUC_100:", RAUC(ideal_rank[:100], muta_graph[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], muta_graph[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], muta_graph[1][2][:1000]))
    #
    # print("DSA")
    # result = dsa_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo, Operator=Operator_train)
    # print("RAUC_100:", RAUC(ideal_rank[:100], result[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], result[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], result[1][2][:1000]))

    # plot_data(path="图/ours.png", data1=curve(ideal_rank), label1="Ideal",
    #           data2=curve(dsa[1][2]), label2="Ours",
    #           Xlabel="Sample", Ylabel="The cumulative number of errors", show=True)
    # print((time()-start)/5)

    # 模型重训练
    # 计算重训练前的精度
    acc = np.sum(np.argmax(model.predict(testset[0]), axis=1) == testset[1]) / len(testset[0])
    print("重训练前的模型精度：{}%".format(acc))
    # 重训练

    sorted_id = result[0][:100]   # 前100个id

    optimizer = optimizers.SGD(lr=0.001)
    # optimizer = optimizers.SGD(lr=.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.summary()

    model.fit(testset[0][sorted_id], testset[1][sorted_id], batch_size=128, epochs=5, shuffle=None, verbose=0)
    # 计算重训练后的精度
    acc_retrain = np.sum(np.argmax(model.predict(testset[0]), axis=1) == testset[1]) / len(testset[0])
    print("重训练后的模型精度：{}%".format(acc_retrain))

    print("end")









