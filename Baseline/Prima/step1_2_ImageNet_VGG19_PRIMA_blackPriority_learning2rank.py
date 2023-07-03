from keras.datasets import cifar10
import numpy as np
from keras.layers import Dense
from keras.models import Sequential,load_model
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from keras.applications.vgg19 import VGG19, decode_predictions
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import cv2
from keras_preprocessing.image import ImageDataGenerator

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import backend as BE
from keras.losses import categorical_crossentropy
from progressbar import ProgressBar
import os
import tensorflow as tf
import json
import sys
import scipy.io
sys.path.append('..')  
from GradPri_utils.utils import *

import xgboost
from sklearn import model_selection

# from keras import backend as BE
# from Integrated_Gradients_algorithm import *
# from GradVisualizer import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess=tf.compat.v1.Session(config=config) 

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()"

DATA_PATH = '../datasets/ImageNetVal/'
file_name = getfile_name(DATA_PATH)
file_name = np.sort(file_name)

f = open("/public/liujiawei/huawei/ZHB/ADF-master/datasets/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",encoding = "utf-8")
val_ground_truth = f.read()
val_ground_truth = val_ground_truth.split('\n')
for i in range(len(val_ground_truth)-1):
    val_ground_truth[i] = int(val_ground_truth[i])

vgg19_json = json.load(open('/public/liujiawei/.keras/models/imagenet_class_index.json','r',encoding="utf-8"))

synsets = scipy.io.loadmat(os.path.join('/public/liujiawei/huawei/ZHB/ADF-master/datasets',
                                        'ILSVRC2012_devkit_t12', 'data', 'meta.mat'))['synsets']

ILSVRC2012_ID = [s[0][0][0][0] for s in synsets]

index1 = 821
WNID = [s[0][1][0] for s in synsets]
print(WNID[index1])

words = [s[0][2][0] for s in synsets]
print(words[index1])

num_train_images = [s[0][7][0][0] for s in synsets]
print(num_train_images[0])

# 预设的参数
total_sample_num = 10000  # 待测的总样本数
img_size = (224,224)
feature_PRIMA = np.load('./featureExtraction/ImageNet_'+str(total_sample_num)+'samplesBlackFeature.npy',
                        allow_pickle=True)

X_xgboost = np.zeros((total_sample_num, (7+10)))
pbar = ProgressBar()
for i in pbar(range(0,total_sample_num)):
    X_xgboost[i, 0] = feature_PRIMA.item()['euler'][i]  # 0范数
    X_xgboost[i, 1] = feature_PRIMA.item()['mahat'][i]  # 2范数
    X_xgboost[i, 2] = feature_PRIMA.item()['qube'][i]   # 无穷范数
    X_xgboost[i, 3] = feature_PRIMA.item()['cos'][i][0] # 余弦
    X_xgboost[i, 4] = feature_PRIMA.item()['difference'][i] # 最大概率差值
    X_xgboost[i, 5] = feature_PRIMA.item()['wnum'][i]   # 变异体分错为其他的类数
    X_xgboost[i, 6] = feature_PRIMA.item()['num_mc'][i] # 变异体分错为其他类的最大数
    X_xgboost[i, 7:17] = feature_PRIMA.item()['fenbu'][i]

    # 用于XGBoost训练的样本的真实标签，即是否为误分类
    Y_xgboost = []
    top_set = 1 # 预测的前n个类中包含真实标签则表示预测正确
    predicted_confidence = feature_PRIMA.item()['predicted_confidence']
    ground_truth_label = feature_PRIMA.item()['ground_truth_label']
    pbar = ProgressBar()
    for i in pbar(range(0, total_sample_num)):
        if top_set is not None:
            if not get_acc(predict_label=decode_predictions(predicted_confidence[i], top=top_set),
                               ground_truth=ground_truth_label[i]):
                Y_xgboost.append(1)
            else:
                Y_xgboost.append(0)

    Y_xgboost = np.array(Y_xgboost)

# X_train_xgboost, X_test_xgboost, y_train_xgboost, y_test_xgboost = model_selection.train_test_split(
#     X_xgboost, Y_xgboost, test_size=0.3)

# xg_reg = xgboost.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
# xg_reg.fit(X_train_xgboost, y_train_xgboost)

# # 预测所有数据的
# y_pred_xgbooxt = xg_reg.predict(X_xgboost)
# indexs = np.argsort(y_pred_xgbooxt)
# indexs = indexs[::-1]
# APFD,_,wrong_index = get_APFD(Gini_indexs=indexs, ground_truth_label=np.array(ground_truth_label),
#                               predicted_confidence=np.array(predicted_confidence), top_set=top_set, decode_predictions=decode_predictions)
# print(\"APFD: \", APFD)
# RAUC,_,_ = get_RAUC(Gini_indexs=indexs, ground_truth_label=np.array(ground_truth_label),
#                 predicted_confidence=np.array(predicted_confidence), top_set=top_set, decode_predictions=decode_predictions)
# print(\"RAUC: \", RAUC)

train_num_xgboost = 2000  # 用于训练的样本数
bottom_train_xgboost = range(total_sample_num-train_num_xgboost, total_sample_num)  # 后bottom_train_num_xgboost个作为训练的样本
xg_reg = xgboost.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_xgboost[bottom_train_xgboost], Y_xgboost[bottom_train_xgboost])

# 预测xgboost的测试集部分
top_test_xgboost = range(0, 1000)  # 前top_test_num_xgboost个作为测试的样本

y_pred_xgbooxt = xg_reg.predict(X_xgboost[top_test_xgboost])
indexs = np.argsort(y_pred_xgbooxt)
indexs = indexs[::-1]
APFD,_,wrong_index = get_APFD(Gini_indexs=indexs, ground_truth_label=np.array(ground_truth_label)[top_test_xgboost],
                              predicted_confidence=np.array(predicted_confidence[top_test_xgboost]),
                              top_set=top_set, decode_predictions=decode_predictions)
print("APFD: ", APFD)
RAUC,_,_ = get_RAUC(Gini_indexs=indexs, ground_truth_label=np.array(ground_truth_label)[top_test_xgboost],
                predicted_confidence=np.array(predicted_confidence[top_test_xgboost]),
                    top_set=top_set, decode_predictions=decode_predictions)
print("RAUC: ", RAUC)



