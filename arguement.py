import numpy as np

from common import *
from Load_dataset import *
from Constant import *
from keras.preprocessing.image import ImageDataGenerator


if __name__ == "__main__":
    #
    Dataset = "Cifar100"
    Mo = "VGG19"
    ope = "Rotate"

    train_at_num = 30000    # 3w个训练集对抗样本
    train_tr_num = 60000-train_at_num
    test_at_num = 2000  # 2k个测试集对抗样本
    test_tr_num = 10000-test_at_num

    # 加载模型
    model = get_model(Dataset=Dataset, Mo=Mo)

    # 加载数据
    X_train, Y_train, _, _, X_test, Y_test =load_dataset(Dataset=Dataset, split=False)

    X_train_ture, Y_train_ture, X_train_false, Y_train_false = filter_sample(model, X_train, Y_train)
    X_test_ture, Y_test_ture, X_test_false, Y_test_false = filter_sample(model, X_test, Y_test)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=4,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

    # 对训练集和测试集攻击
    generator_train = datagen.flow(X_train_ture, Y_train_ture, batch_size=len(X_train_ture))
    X_train_rotate = generator_train[0][0]
    Y_train_rotate = generator_train[0][1]
    _, _, X_train_rotate_false, Y_train_rotate_false = filter_sample(model, X_train_rotate, Y_train_rotate)

    generator_test = datagen.flow(X_test_ture, Y_test_ture, batch_size=len(X_test_ture))
    X_test_rotate = generator_test[0][0]
    Y_test_rotate = generator_test[0][1]
    _, _, X_test_rotate_false, Y_test_rotate_false = filter_sample(model, X_test_rotate, Y_test_rotate)

    # 重组
    X_train = np.concatenate((X_train_ture[:train_tr_num],
                              np.concatenate((X_train_rotate_false, X_train_rotate_false, X_train_rotate_false), axis=0)[:train_at_num]),
                             axis=0)
    Y_train = np.concatenate((Y_train_ture[:train_tr_num],
                              np.concatenate((Y_train_rotate_false, Y_train_rotate_false, Y_train_rotate_false), axis=0)[:train_at_num]),
                             axis=0)
    X_test = np.concatenate((np.concatenate((X_test_ture, X_test_ture), axis=0)[:test_tr_num],
                             np.concatenate((X_test_rotate_false, X_test_rotate_false, X_test_rotate_false), axis=0)[:test_at_num]),
                            axis=0)
    Y_test = np.concatenate((np.concatenate((Y_test_ture, Y_test_ture), axis=0)[:test_tr_num],
                             np.concatenate((Y_test_rotate_false, Y_test_rotate_false, Y_test_rotate_false), axis=0)[:test_at_num]),
                            axis=0)
    print(model.evaluate(X_train, Y_train))
    print(model.evaluate(X_test, Y_test))

    # 打乱
    train_idx = np.random.choice(len(X_train), len(X_train), replace=False)
    test_idx = np.random.choice(len(X_test), len(X_test), replace=False)

    X_train = X_train[train_idx]
    Y_train = Y_train[train_idx]
    X_test = X_test[test_idx]
    Y_test = Y_test[test_idx]

    # 保存
    np.savez("Adv/{}/{}/{}_{}.npz".format(Dataset, Mo, Dataset, ope),
             X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    print("end")


