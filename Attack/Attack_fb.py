import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径

import foolbox as fb
from tqdm import tqdm


class fb_attack():
    def __init__(self, model, ope):
        self.model = model  # 原模型
        self.fmodel = fb.models.KerasModel(self.model, bounds=(0, 1))   # fmodel
        self.ope = ope      # 攻击方法
        self.attack = None
        if self.ope == "CW":
            self.attack = fb.attacks.CarliniWagnerL2Attack(self.fmodel)
        elif self.ope == "PGD":
            self.attack = fb.attacks.PGD(self.fmodel)
        elif self.ope == "JSMA":
            self.attack = fb.attacks.SaliencyMapAttack(self.fmodel)
        elif self.ope == "FGSM":
            self.attack = fb.attacks.FGSM(self.fmodel)

    def generate_adv(self, X, Y, attsize):
        adv_sucess = [] # 保存攻击成功的对抗样本
        adv_y = []      # 对抗样本对应类标
        cnt = 0
        for idx, x in tqdm(enumerate(X)):
            y = Y[idx]  # 真实类标
            if np.argmax(self.model.predict(np.array([x]))) != y:
                continue    # 正常样本预测失败不攻击
            # 开始攻击
            if self.ope == "CW":
                adv = self.attack(x, label=y)  # 生成对抗样本
            elif self.ope == "PGD":
                adv = self.attack(x, label=y)
            elif self.ope == "FGSM":
                adv = self.attack(x, label=y)
            elif self.ope == "JSMA":
                adv = self.attack(x, label=y)

            if np.sum(adv)!=None:
                ad_label = np.argmax(self.model.predict(np.expand_dims(adv, axis=0)))   # 预测对抗样本
                if ad_label != y:   # 样本攻击成功
                    adv_sucess.append(adv)
                    adv_y.append(y)
                    cnt += 1
            if cnt == attsize:
                break
        adv_sucess = np.array(adv_sucess)
        adv_y = np.array(adv_y)
        return adv_sucess, adv_y


if __name__ == "__main__":
    print("end")







