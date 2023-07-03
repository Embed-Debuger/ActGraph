# import tensorflow.compat.v1 as tf
import os
import time
import numpy as np
from Model_Lenet5 import *
from Model_vgg import *
from Model_Resnet import *
from Load_dataset import load_dataset
from Constant import *
# from Draw import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


def lenet5_train_cifar10():
    # 加载模型
    model = lenet5_cifar10(lr=0.01, drop_rate=0.3)
    # 加载数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10(onehot=False, sequence=False, split=True)

    # 创建文件夹
    localtime = time.asctime(time.localtime(time.time())).replace(" ", "-")  # 读取当前时间
    weight_path = "Model/Cifar10/Lenet5/"  # 时间作为文件夹名
    os.makedirs(weight_path, exist_ok=True)

    model_checkpoint = ModelCheckpoint(os.path.join(weight_path, localtime + ".h5"),
                                       monitor="val_loss", save_best_only=True, save_weights_only=False,
                                       verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    his = model.fit(X_train, Y_train, batch_size=128, epochs=100, shuffle=None,
                    callbacks=[model_checkpoint, earlystop],
                    validation_data=(X_val, Y_val), verbose=1)  # 训练模型 模型直接更新到model 返回history记录训练过程


def vgg11_train_cifar10():
    # 加载模型
    model = vgg11(lr=0.01, drop_rate=0.5, path="Model/Cifar10/VGG11/Thu-May--5-19:30:44-2022.h5")
    # 加载数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10(onehot=True, sequence=False, split=True)

    # 创建文件夹
    localtime = time.asctime(time.localtime(time.time())).replace(" ", "-")  # 读取当前时间
    weight_path = "Model/Cifar10/VGG11/"    # 时间作为文件夹名
    os.makedirs(weight_path, exist_ok=True)

    model_checkpoint = ModelCheckpoint(os.path.join(weight_path, localtime+".h5"),
                                       monitor="val_loss", save_best_only=True, save_weights_only=False,
                                       verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    his = model.fit(X_train, Y_train, batch_size=128, epochs=100, shuffle=None,
                    callbacks=[model_checkpoint, earlystop],
                    validation_data=(X_val, Y_val), verbose=1)  # 训练模型 模型直接更新到model 返回history记录训练过程

    print(model.evaluate(X_test, Y_test))


def lenet5_train_mnist():
    # 加载模型
    model = lenet5_mnist(lr=0.01, drop_rate=0.5, path="../Model_weight/Mnist")
    # 加载数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist(onehot=False, sequence=False, split=True)

    # 创建文件夹
    localtime = time.asctime(time.localtime(time.time())).replace(" ", "-")  # 读取当前时间
    weight_path = "../Model_weight/Mnist/Lenet5"    # 时间作为文件夹名
    os.makedirs(weight_path, exist_ok=True)

    model_checkpoint = ModelCheckpoint(os.path.join(weight_path, localtime+".h5"),
                                       monitor="val_loss", save_best_only=True, save_weights_only=False,
                                       verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    his = model.fit(X_train, Y_train, batch_size=128, epochs=100, shuffle=None,
                    callbacks=[model_checkpoint, earlystop],
                    validation_data=(X_val, Y_val), verbose=1)  # 训练模型 模型直接更新到model 返回history记录训练过程

    print(model.evaluate(X_test, Y_test))


def vgg16_train_cifar10():
    lr = 1e-1

    def lr_scheduler(epoch):
        return lr * (0.1 ** (epoch // 50))

    # 加载模型
    model = VGGNet(classes=10, input_shape=(32, 32, 3),
                   weight_decay=5e-4, conv_block_num=4, fc_layers=2, fc_units=256)
    # sgd
    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # 加载数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10(onehot=False, sequence=False, split=True)

    # 创建文件夹
    localtime = time.asctime(time.localtime(time.time())).replace(" ", "-")  # 读取当前时间
    weight_path = "../Model_weight/Cifar10/VGG16/"  # 时间作为文件夹名
    os.makedirs(weight_path, exist_ok=True)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )
    model_checkpoint = ModelCheckpoint(os.path.join(weight_path, localtime + ".h5"),
                                       monitor="val_loss", save_best_only=True, save_weights_only=False,
                                       verbose=1)
    Reduce = LearningRateScheduler(lr_scheduler)
    model.fit_generator(generator=datagen.flow(X_train, Y_train, batch_size=128),
                        epochs=200,
                        callbacks=[model_checkpoint, Reduce],
                        validation_data=(X_val, Y_val)
                        )

    print(model.evaluate(X_test, Y_test))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    lr = 1e-1
    def lr_scheduler(epoch):
        return lr * (0.1 ** (epoch // 50))
    Reduce = LearningRateScheduler(lr_scheduler)
    # 加载模型
    # model = vgg19()
    # model = VGGNet(classes=10, input_shape=(32, 32, 3),
    #              weight_decay=5e-4, conv_block_num=4, fc_layers=2, fc_units=256)
    model = VGG19Net(classes=100, input_shape=(32, 32, 3),
                   weight_decay=5e-4, conv_block_num=4, fc_layers=2, fc_units=512)

    # sgd
    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    # 加载数据
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10(onehot=False, sequence=False, split=True)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(Dataset="Cifar100", onehot=False, sequence=False, split=True)

    # 创建文件夹
    localtime = time.asctime(time.localtime(time.time())).replace(" ", "-")  # 读取当前时间
    weight_path = "../Model_weight/Cifar100/VGG19/"    # 时间作为文件夹名
    os.makedirs(weight_path, exist_ok=True)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )
    model_checkpoint = ModelCheckpoint(os.path.join(weight_path, localtime+".h5"),
                                       monitor="val_loss", save_best_only=True, save_weights_only=False,
                                       verbose=1)
    # earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    # Reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1,
    #                            mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    model.fit_generator(generator=datagen.flow(X_train, Y_train, batch_size=128),
                      epochs=400,
                      callbacks=[model_checkpoint, Reduce],
                      validation_data=(X_val, Y_val)
                      )
    # his = model.fit(X_train, Y_train, batch_size=128, epochs=200, shuffle=True,
    #                 callbacks=[model_checkpoint, earlystop, Reduce],
    #                 validation_data=(X_val, Y_val), verbose=1)  # 训练模型 模型直接更新到model 返回history记录训练过程

    print(model.evaluate(X_test, Y_test))

    print("end")



