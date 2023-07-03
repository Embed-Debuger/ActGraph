import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径

import foolbox as fb
from tqdm import tqdm

from Model.Model_Lenet5 import *
from Model.Model_vgg import *
from Load_dataset import *


def Boundary_Attack(model, fmodel, example, label):
    # 每次只攻击一个样本
    sucess = 0

    attack = fb.attacks.BoundaryAttack(fmodel)
    ad_siganl = attack(example, label=label, iterations=3000, verbose=False)  # 最大的迭代步长
    # ad_label = np.argmax(model.predict(np.expand_dims(ad_siganl, axis=0)))

    if np.sum(ad_siganl) != None:
        ad_label = np.argmax(model.predict(np.expand_dims(ad_siganl, axis=0)))

        if ad_label != label:
            print("样本攻击成功")
            ad_label = np.argmax(model.predict(np.expand_dims(ad_siganl, axis=0)))
            sucess = 1
        else:
            print("样本攻击失败")
            ad_siganl = example
            ad_label = label
            sucess = 0
    else:
        print("样本攻击失败")
        ad_siganl = example
        ad_label = label
        sucess = 0

    return ad_siganl, ad_label, sucess


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    # 加载模型
    # model = lenet5_cifar10(path=os.path.join(base_path, "../Model_weight/Cifar10/Lenet5/Wed-Feb-23-20:33:29-2022.h5"))
    model = vgg11(path=os.path.join(base_path, "../Model_weight/Cifar10/VGG11/Thu-May--5-21:08:30-2022.h5"))

    # 加载数据集
    X_train, Y_train, _, _, X_test, Y_test = load_cifar10(onehot=False, sequence=False, split=False)

    # 筛选数据集 筛选能正确识别的样本
    X_train, Y_train, _, _ = filter_sample(model, X_train, Y_train)
    X_test, Y_test, _, _ = filter_sample(model, X_test, Y_test)

    fmodel = fb.models.KerasModel(model, bounds=(0, 1))  # foolbox生成fmodel

    # 生成对抗样本
    ADV_NUM = 4000
    # 攻击训练集
    sucess_cnt = 0
    X_train_adv = []
    X_train_clean = []
    train_label = []

    for idx, x in tqdm(enumerate(X_train)):
        y = Y_train[idx]
        ad_image, ad_label, success = Boundary_Attack(model=model, fmodel=fmodel, example=x, label=y)
        if success == 1:
            sucess_cnt += 1
            X_train_adv.append(ad_image)
            X_train_clean.append(x)
            train_label.append(y)
        if sucess_cnt >= ADV_NUM:
            break
    X_train_clean = np.array(X_train_clean)
    X_train_adv = np.array(X_train_adv)
    train_label = np.array(train_label)

    print("训练集干净样本精度：", model.evaluate(X_train_clean, train_label))
    print("训练集对抗样本精度：", model.evaluate(X_train_adv, train_label))

    # 攻击测试集
    sucess_cnt = 0
    X_test_adv = []
    X_test_clean = []
    test_label = []

    for idx, x in tqdm(enumerate(X_test)):
        y = Y_test[idx]
        ad_image, ad_label, success = Boundary_Attack(model=model, fmodel=fmodel, example=x, label=y)
        if success == 1:
            sucess_cnt += 1
            X_test_adv.append(ad_image)
            X_test_clean.append(x)
            test_label.append(y)
        if sucess_cnt >= ADV_NUM:
            break
    X_test_clean = np.array(X_test_clean)
    X_test_adv = np.array(X_test_adv)
    test_label = np.array(test_label)

    print("测试集干净样本精度：", model.evaluate(X_test_clean, test_label))
    print("测试集对抗样本精度：", model.evaluate(X_test_adv, test_label))

    # 保存
    # np.savez(os.path.join(base_path, "../Adv/Cifar10/Lenet5/Boundary_perclass={:d}.npz".format(int(ADV_NUM / 10))),
    #          X_train_clean=X_train_clean, X_train_adv=X_train_adv, train_label=train_label,
    #          X_test_clean=X_test_clean, X_test_adv=X_test_adv, test_label=test_label)

    np.savez(os.path.join(base_path, "../Adv/Cifar10/VGG11/Boundary_perclass={:d}.npz".format(int(ADV_NUM/10))),
             X_train_clean=X_train_clean, X_train_adv=X_train_adv, train_label=train_label,
             X_test_clean=X_test_clean, X_test_adv=X_test_adv, test_label=test_label)

    print("end")



