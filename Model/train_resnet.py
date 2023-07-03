import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import losses
from keras import optimizers
import tensorflow as tf
from keras import backend as K
from Model_Resnet import *
import os
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import time


class CIFAR10Solver(object):
    """
    A CIFAR10Solver encapsulates all the logic nessary for training cifar10 classifiers.The train model is defined
    outside, you must pass it to init().
    The solver train the model, plot loss and aac history, and test on the test data.
    Example usage might look something like this.
    model = MyAwesomeModel(opt=SGD, losses='categorical_crossentropy',  metrics=['acc'])
    model.compile(...)
    model.summary()
    solver = CIFAR10Solver(model)
    history = solver.train()
    plotHistory(history)
    solver.test()
    """

    def __init__(self, model, data):
        """
        :param model: A model object conforming to the API described above
        :param data:  A tuple of training, validation and test data from CIFAR10Data
        """
        self.model = model
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = data

    def __on_epoch_end(self, epoch, logs=None):
        print(K.eval(self.model.optimizer.lr))

    def train(self, epochs=200, batch_size=128, data_augmentation=True, callbacks=None):
        if data_augmentation:
            # datagen
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=4,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
            )
            # (std, mean, and principal components if ZCA whitening is applied).
            # datagen.fit(x_train)
            print('train with data augmentation')
            train_gen = datagen.flow(self.X_train, self.Y_train, batch_size=batch_size)
            history = self.model.fit_generator(generator=train_gen,
                                               epochs=epochs,
                                               callbacks=callbacks,
                                               validation_data=(self.X_val, self.Y_val)
                                               # steps_per_epoch=int(self.X_train.shape[0]/batch_size)
                                               )
        else:
            print('train without data augmentation')
            history = self.model.fit(self.X_train, self.Y_train,
                                     batch_size=batch_size, epochs=epochs,
                                     callbacks=callbacks,
                                     validation_data=(self.X_val, self.Y_val),
                                     )
        return history

    def test(self):
        loss, acc = self.model.evaluate(self.X_test, self.Y_test)
        print('test data loss:%.2f acc:%.4f' % (loss, acc))


if __name__ == "__main__":
    from Load_dataset import *

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # 创建文件夹
    localtime = time.asctime(time.localtime(time.time())).replace(" ", "-")  # 读取当前时间
    weight_path = "../Model_weight/Cifar10/Resnet18/"    # 时间作为文件夹名
    os.makedirs(weight_path, exist_ok=True)

    # 加载数据
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10(onehot=False, sequence=False, split=True)

    weight_decay = 1e-4
    lr = 1e-1
    num_classes = 10
    model = ResNet18(input_shape=(32, 32, 3), classes=num_classes, weight_decay=weight_decay)
    opt = optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
    model.compile(optimizer=opt,
                  loss=losses.sparse_categorical_crossentropy,
                  # loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    def lr_scheduler(epoch):
        new_lr = lr * (0.1 ** (epoch // 50))
        print('new lr:%.2e' % new_lr)
        return new_lr

    reduce_lr = LearningRateScheduler(lr_scheduler)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_path, localtime+".h5"),
                                       monitor="val_loss", save_best_only=True, save_weights_only=False,
                                       verbose=1)
    solver = CIFAR10Solver(model, (X_train, Y_train, X_val, Y_val, X_test, Y_test))
    solver.train(epochs=200, batch_size=64, data_augmentation=True, callbacks=[model_checkpoint, reduce_lr])
    model.evaluate(X_test, Y_test)


