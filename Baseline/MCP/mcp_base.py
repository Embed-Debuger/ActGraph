import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径
import numpy as np
from common import *

def mcp_rank(model, testset, selectsize, ideal_rank):
    X_test = testset[0]     # 图像
    Y_test = testset[1]     # 类标
    bug_test = testset[2]   # bug标

    act_layers = model.predict(X_test, batch_size=32)  # 模型置信度输出
    dicratio = [[] for i in range(100)]
    dicindex = [[] for i in range(100)]
    # 初始化存储空间
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act)  # max_index
        dicratio[max_index * 10 + sec_index].append(ratio)  # 保存各类值
        dicindex[max_index * 10 + sec_index].append(i)

    selected_lst = select_from_firstsec_dic2(selectsize, dicratio, dicindex)

    x = X_test[selected_lst]
    y = Y_test[selected_lst]
    bug = bug_test[selected_lst]

    # 计算指标
    apfd_all = RAUC(ideal_rank, bug)
    print("APFD分数：%.3f%%" % (apfd_all * 100))

    return selected_lst, (x, y, bug)


def find_second(act):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(10):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(10):
        if i == max_index:
            continue
        if act[i] > second_max:
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_  # 第二大的值除以第一大的值
    # print 'max:',max_index
    return max_index, sec_index, ratio


def select_from_firstsec_dic(selectsize, dicratio, dicindex):
    selected_lst = []
    tmpsize = selectsize

    noempty = no_empty_number(dicratio)
    # print(selectsize)
    # print(noempty)
    while selectsize >= noempty:
        for i in range(100):
            if len(dicratio[i]) != 0:
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                if tmp >= 0.1:
                    selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)

    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]
        max_index_tmp = [0 for i in range(selectsize)]
        for i in range(100):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    # selected_lst.append()
                    # if tmp_max>=0.1:
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
    # print(selected_lst)
    assert len(selected_lst) == tmpsize
    return selected_lst


def select_from_firstsec_dic2(selectsize, dicratio, dicindex):
    select_ls = []
    while selectsize != len(select_ls):
        for i in range(len(dicratio)):
            if dicratio[i] != []:  # 不为空
                temp = np.array(dicratio[i])  # 转为numpy方便计算
                max_idx = np.argmax(temp)
                max_v = dicratio[i][max_idx]
                select_ls.append(dicindex[i][max_idx])
                del dicratio[i][max_idx]
                del dicindex[i][max_idx]
            if len(select_ls) == selectsize:
                break
    assert len(select_ls) == selectsize
    return select_ls


def no_empty_number(dicratio):
    # 计算列表非零数
    no_empty=0
    for i in range(len(dicratio)):
        if len(dicratio[i])!=0:
            no_empty+=1
    return no_empty


def high_number(dicratio):
    # 计算列表大于阈值的数
    num=0
    thres = 0
    for i in range(len(dicratio)):
        num += np.sum(np.array(dicratio[i]) >= thres)
    return num


if __name__ == "__main__":

    print("end")

