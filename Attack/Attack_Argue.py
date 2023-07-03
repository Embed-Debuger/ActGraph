from keras.preprocessing.image import ImageDataGenerator

from Model.Model_Lenet5 import *


def Data_argument(x, y, shuffle=True):
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True)
    datagen.fit(x, augment=True, rounds=1)
    argue_result = datagen.flow(x, y, batch_size=len(x), shuffle=shuffle)
    x_argue = argue_result[0][0]
    x_argue = (x_argue - np.min(x_argue)) / (np.max(x_argue) - np.min(x_argue))    # 归一化
    y_argue = argue_result[0][1]
    return x_argue, y_argue


if __name__ == "__main__":
    # 生成数据增强样本
    # 加载数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10(onehot=False, sequence=False, split=False)

    # 数据增强
    X_train_argue, Y_train_argue = Data_argument(X_train, Y_train, shuffle=False)
    X_test_argue, Y_test_argue = Data_argument(X_test, Y_test, shuffle=False)

    # 加载模型
    model = lenet5_cifar10(path="Model/Cifar10/Lenet5/Wed-Feb-23-20:33:29-2022.h5")
    model.evaluate(X_train_argue, Y_train_argue)
    model.evaluate(X_test_argue, Y_test_argue)

    # np.savez("Dataset/cifar10_argue", X_train=X_train_argue, Y_train=Y_train_argue,
    #          X_test=X_test_argue, Y_test=Y_test_argue)
    plt.imshow(X_test_argue[0])
    plt.show()
    plt.close()
    print("end")


