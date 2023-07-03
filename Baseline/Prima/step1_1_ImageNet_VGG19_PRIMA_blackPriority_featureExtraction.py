from keras.datasets import cifar10
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19, decode_predictions
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from keras_preprocessing.image import ImageDataGenerator
import cv2

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
# from tensorflow.keras import backend as BE
# from Integrated_Gradients_algorithm import *
# from GradVisualizer import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess=tf.compat.v1.Session(config=config)

DATA_PATH = '../datasets/ImageNetVal/'
file_name = getfile_name(DATA_PATH)
file_name = np.sort(file_name)

f = open("/public/liujiawei/huawei/ZHB/ADF-master/datasets/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",encoding = "utf-8")

val_ground_truth = f.read()
val_ground_truth = val_ground_truth.split('\n')
for i in range(len(val_ground_truth)-1):
    val_ground_truth[i] = int(val_ground_truth[i])

vgg19_json = json.load(open('/public/liujiawei/.keras/models/imagenet_class_index.json','r',encoding="utf-8"))
base_model = VGG19(weights='imagenet')

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

base_model.summary()
# 预设的参数
total_sample_num = 10000  # 待测的总样本数
sample_mutant_num = 200 # 单个样本得到的变异体数量，PRIMA中设置图像为200个，文本为50个
img_size = (224,224)

pbar = ProgressBar()

id_ = []
euler_ = []
mahat_ = []
qube_ = []
cos_ = []
difference_ = []
wrong_class_num_ = []
max_class_num_ = []
cos_dis_ = []

predicted_confidence = []
ground_truth_label = []

for i in pbar(range(0, total_sample_num)):
    img_path = DATA_PATH + file_name[i]
    img = image.load_img(img_path, target_size=img_size)
    x_tmp = image.img_to_array(img)
    x_tmp = np.expand_dims(x_tmp, axis=0)
    x_tmp = preprocess_input(x_tmp)
    pre_tmp = base_model.predict(np.reshape(x_tmp, [-1,img_size[0],img_size[1],3]))

    a = pre_tmp  #ori_prob[i] # 样本的预测置信度
    max_value = np.max(a)  # 预测置信度的最大概率值
    max_value_pos = np.argmax(a)   # 样本的预测标签
#     perturbated_prediction   # 样本扰动后的预测置信度
    # 对于待测样本的变异扰动样本的特征提取
    euler = 0
    mahat = 0
    qube = 0
    cos = 0
    difference = 0
    different_class = []
    cos_list = []

    perturbated_img = []
    for mutants_input in range(0, int(sample_mutant_num/5)):  # 循环sample_mutant_num次，生成sample_mutant_num个变异样本
        row1 = np.random.randint(0, img_size[0], dtype=int)  # 确定扰动的像素纵坐标（行）
        col1 = np.random.randint(0, img_size[1], dtype=int)  # 确定扰动的像素横坐标（列）
        tt = gauss_noise(x_tmp,row1,col1,ratio=1.0,var=0.01)
        perturbated_img.append(tt)
        tt = white(x_tmp,row1,col1)
        perturbated_img.append(tt)
        tt = black(x_tmp,row1,col1)
        perturbated_img.append(tt)
        tt = reverse_color(x_tmp,row1,col1)
        perturbated_img.append(tt)
        tt = shuffle_pixel(x_tmp,row1,col1)
        perturbated_img.append(tt)
    perturbated_img = np.squeeze(np.array(perturbated_img), axis=1)
    perturbated_prediction = base_model.predict(perturbated_img)  # 样本扰动后的预测置信度

    for pp in perturbated_prediction:  # 取每一种变异扰动样本的置信度值
        pro = pp
        opro = a
        # if np.argmax(ii) != result[i]:
        difference += abs(max_value - pp[max_value_pos])
        euler += np.linalg.norm(pro - opro)     # 正常样本对变异样本的预测概率范数
        mahat += np.linalg.norm(pro - opro, ord=1)
        qube += np.linalg.norm(pro - opro, ord=np.inf)
        co = (1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))))
        if co < 0:
            co = 0
        elif co > 1:
            co = 1
        cos += co
        cos_list.append(co)
        if np.argmax(pp) != max_value_pos:  # 如果变异后结果不一样
            different_class.append(np.argmax(pp))   # 保存变异类的预测类标
    cos_dis = cos_distribution(cos_list)
    # euler /= 256
    # mahat /= 256
    # qube /= 256
    # cos /= 256
    dic = {}
    for key in different_class:
        dic[key] = dic.get(key, 0) + 1
    wrong_class_num = len(dic)
    if len(dic) > 0:
        max_class_num = max(dic.values())
    else:
        max_class_num = 0

    id_.append(i)
    euler_.append(euler)
    mahat_.append(mahat)
    qube_.append(qube)
    cos_.append(cos)
    difference_.append(difference)
    wrong_class_num_.append(wrong_class_num)
    max_class_num_.append(max_class_num)
    cos_dis_.append(cos_dis)
    predicted_confidence.append(pre_tmp)
    ground_truth_label.append(WNID[val_ground_truth[i]-1])
#     print('id:',i)
#     print('euler:', euler)
#     print('mahat:', mahat)
#     print('qube:', qube)
#     print('cos:', cos)
#     print('difference:',difference)
#     print('wnum:',wrong_class_num)
#     print('num_mc:', max_class_num)
#     print('fenbu:',cos_dis)"

id_ = np.array(id_)
euler_ = np.array(euler_)
mahat_ = np.array(mahat_)
qube_ = np.array(qube_)
cos_ = np.array(cos_)
difference_ = np.array(difference_)
wrong_class_num_ = np.array(wrong_class_num_)
max_class_num_ = np.array(max_class_num_)
cos_dis_ = np.array(cos_dis_)
predicted_confidence = np.array(predicted_confidence)

np.save('./featureExtraction/ImageNet_'+str(total_sample_num)+'samplesBlackFeature.npy',{
    'id': id_,
    'euler': euler_,
    'mahat': mahat_,
    'qube': qube_,
    'cos': cos_,
    'difference': difference_,
    'wnum': wrong_class_num_,
    'num_mc': max_class_num_,
    'fenbu': cos_dis_,
    'predicted_confidence': predicted_confidence,
    'ground_truth_label': ground_truth_label,
})


if __name__ == "__main__":
    print("end")

