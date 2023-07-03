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


if __name__ == "__main__":
    Dataset = "Mnist"
    Mo = "Lenet5"
    Operator = "Rotate"
    train_num = 500
    test_num = 1000

    print(Dataset, Mo, Operator, "train_num:", train_num, "test_num:", test_num)
    model = get_model(Dataset, Mo)
    X_train, Y_train, X_test, Y_test = get_dataset(Dataset, Mo, Operator)

    # print(model.evaluate(X_train, Y_train))
    # print(model.evaluate(X_test, Y_test))

    bug_train = np.where(np.argmax(model.predict(X_train), axis=1) == Y_train, 0, 1)  # 样本是否触发模型漏洞
    bug_test = np.where(np.argmax(model.predict(X_test), axis=1) == Y_test, 0, 1)  # 样本是否触发模型漏洞

    trainset = (X_train[:train_num], Y_train[:train_num], bug_train[:train_num])
    testset = (X_test[:test_num], Y_test[:test_num], bug_test[:test_num])
    ideal_rank = np.sort(bug_test[:test_num])[::-1]    # 测试用例的理想曲线

    # print("DeepGini")
    # gini = gini_rank(model, testset, ideal_rank)
    # print("RAUC_100:", RAUC(ideal_rank[:100], gini[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], gini[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], gini[1][2][:1000]))
    #
    # # print("SRS")
    # # srs = srs_rank(testset, ideal_rank)
    # # print("RAUC_100:", RAUC(ideal_rank[:100], srs[1][2][:100]))
    # # print("RAUC_500:", RAUC(ideal_rank[:500], srs[1][2][:500]))
    # # print("RAUC_1000:", RAUC(ideal_rank[:1000], srs[1][2][:1000]))
    #
    print("激活")
    act = act_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo)
    print("RAUC_100:", RAUC(ideal_rank[:100], act[1][2][:100]))
    print("RAUC_500:", RAUC(ideal_rank[:500], act[1][2][:500]))
    print("RAUC_1000:", RAUC(ideal_rank[:1000], act[1][2][:1000]))
    #
    # print("MCP")
    # mcp = mcp_rank(model, testset, selectsize=len(testset[0]), ideal_rank=ideal_rank)
    # print("RAUC_100:", RAUC(ideal_rank[:100], mcp[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], mcp[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], mcp[1][2][:1000]))
    #
    # print("Prima")
    # prima = prima_model_rank(model, trainset, testset, ideal_rank=ideal_rank, Dataset=Dataset, Mo=Mo)
    # print("RAUC_100:", RAUC(ideal_rank[:100], prima[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], prima[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], prima[1][2][:1000]))
    #
    # print("Graph")
    # graph = graph_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo)
    # print("RAUC_100:", RAUC(ideal_rank[:100], graph[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], graph[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], graph[1][2][:1000]))

    # print("Muta_Graph")
    # muta_graph = mutation_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo)
    # print("RAUC_100:", RAUC(ideal_rank[:100], muta_graph[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], muta_graph[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], muta_graph[1][2][:1000]))

    # print("DSA")
    # dsa = dsa_rank(model, trainset, testset, ideal_rank, Dataset=Dataset, Mo=Mo, Operator=Operator)
    # print("RAUC_100:", RAUC(ideal_rank[:100], dsa[1][2][:100]))
    # print("RAUC_500:", RAUC(ideal_rank[:500], dsa[1][2][:500]))
    # print("RAUC_1000:", RAUC(ideal_rank[:1000], dsa[1][2][:1000]))

    # plot_data(path="图/ours.png", data1=curve(ideal_rank), label1="Ideal",
    #           data2=curve(dsa[1][2]), label2="Ours",
    #           Xlabel="Sample", Ylabel="The cumulative number of errors", show=True)
    print("end")









