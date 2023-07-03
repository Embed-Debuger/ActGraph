from common import *
from Constant import *


if __name__ == "__main__":
    Dataset = "Cifar100"
    Mo = "VGG19"
    Operator = "Mix-JCR"
    Operators = ["origin", "JSMA", "CW", "Rotate"]
    train_num = 60000
    test_num = 10000

    model = get_model(Dataset, Mo)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for ope in Operators:
        X_train_tmp, Y_train_tmp, X_test_tmp, Y_test_tmp = get_dataset(Dataset, Mo, ope)
        X_train.append(X_train_tmp)
        Y_train.append(Y_train_tmp)
        X_test.append(X_test_tmp)
        Y_test.append(Y_test_tmp)

    if Dataset=="Mnist":
        X_train = np.array(X_train).reshape(-1, 28, 28, 1)
        Y_train = np.array(Y_train).reshape(-1, )
        X_test = np.array(X_test).reshape(-1, 28, 28, 1)
        Y_test = np.array(Y_test).reshape(-1, )
    else:
        X_train = np.array(X_train).reshape(-1, 32, 32, 3)
        Y_train = np.array(Y_train).reshape(-1, )
        # X_test = np.array(X_test).reshape(-1, 32, 32, 3)
        # Y_test = np.array(Y_test).reshape(-1, )
        X_test = np.vstack((X_test[0], X_test[1], X_test[2], X_test[3]))
        Y_test = np.hstack((Y_test[0], Y_test[1], Y_test[2], Y_test[3]))

    idx_train = np.random.choice(len(X_train), train_num, replace=False)
    idx_test = np.random.choice(len(X_test), test_num, replace=False)

    X_train = X_train[idx_train]
    Y_train = Y_train[idx_train]
    X_test = X_test[idx_test]
    Y_test = Y_test[idx_test]

    print(model.evaluate(X_train, Y_train))
    print(model.evaluate(X_test, Y_test))
    np.savez("Adv/{}/{}/{}_{}.npz".format(Dataset, Mo, Dataset, Operator),
             X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print("end")



