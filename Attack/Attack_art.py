from art.estimators.classification import KerasClassifier
from art.attacks.evasion import SaliencyMapMethod


class art_attack():
    def __init__(self, model, ope):
        self.model = model
        self.fmodel = KerasClassifier(clip_values=(0, 1), model=self.model, preprocessing=(0, 1))
        self.attack = None
        self.ope = ope
        if self.ope == "JSMA":
            self.attack = SaliencyMapMethod(self.fmodel, verbose=True)

    def generate_adv(self, X, Y, attsize):
        adv = self.attack.generate(X, Y)[:attsize]
        return adv


if __name__ == "__main__":
    print("end")


