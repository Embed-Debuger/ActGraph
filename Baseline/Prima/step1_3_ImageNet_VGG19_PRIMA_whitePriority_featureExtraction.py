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
from keras.models import clone_model

# from keras import backend as BE
# from Integrated_Gradients_algorithm import *
# from GradVisualizer import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess=tf.compat.v1.Session(config=config) 

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

DATA_PATH = '../datasets/ImageNetVal/'
file_name = getfile_name(DATA_PATH)
file_name = np.sort(file_name)

f = open("/public/liujiawei/huawei/ZHB/ADF-master/datasets/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",encoding = "utf-8")
val_ground_truth = f.read()
val_ground_truth = val_ground_truth.split('\\n')
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
total_sample_num = 3000  # 待测的总样本数
model_mutant_num = 10 # 单个模型得到的变异体数量，PRIMA中设置变异模型为100个
weight_mutant_rate = 0.1  # 模型中的权重修改比例，，PRIMA中设置为10%
img_size = (224,224)
layer_pos = 24  # 修改的层的位置"

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

# 获取模型某一层的激活输出
layer_name = 'fc1'
hidden_layer_Model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

# 搭建部分的模型
Input_hidden = Input(shape=(np.shape(base_model.get_layer(layer_name).output[0])))
partial_model_Model = Input_hidden
layer_pos = getLayerIndexByName(base_model, layer_name)
for layer in base_model.layers[layer_pos+1:]:
    partial_model_Model = layer(partial_model_Model)
partial_model_Model = Model(inputs=Input_hidden, outputs=partial_model_Model)


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

    # 进行模型的变异操作
    perturbated_prediction = []
    for ii in range(0, int(model_mutant_num/4)): # 有四种模型变异操作
        conf_NAI = NeuActInverse_confidence(x_tmp, hidden_layer_Model,
                                            partial_model_Model, weight_mutant_rate)  # 神经元激活翻转
        conf_NEB = NeuEffBlock_confidence(x_tmp, hidden_layer_Model,
                                          partial_model_Model, weight_mutant_rate)  # 神经元激活置零
        conf_GF = GaussFuzz_confidence(x_tmp, base_model, 0, 0.1, layer_name)  # 神经元激活添加高斯噪声
        conf_WS = WeightShuffl_confidence(x_tmp, base_model, layer_name, weight_mutant_rate)  # 部分权重值打乱
        perturbated_prediction.append(conf_NAI)
        perturbated_prediction.append(conf_NEB)
        perturbated_prediction.append(conf_GF)
        perturbated_prediction.append(conf_WS)
    perturbated_prediction = np.squeeze(np.array(perturbated_prediction), axis=1)

    for pp in perturbated_prediction:  # 取每一种变异扰动样本的置信度值
        pro = pp
        opro = a
        # if np.argmax(ii) != result[i]:
        difference += abs(max_value - pp[max_value_pos])
        euler += np.linalg.norm(pro - opro)
        mahat += np.linalg.norm(pro - opro, ord=1)
        qube += np.linalg.norm(pro - opro, ord=np.inf)
        co = (1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))))
        if co < 0:
            co = 0
        elif co > 1:
            co = 1
        cos += co
        cos_list.append(co)
        if np.argmax(pp) != max_value_pos:
            different_class.append(np.argmax(pp))   # 变异后的样本错分为其他类
    cos_dis = cos_distribution(cos_list)
    dic = {}
    for key in different_class:
        dic[key] = dic.get(key, 0) + 1  # 统计被错分成哪几类样本
    wrong_class_num = len(dic)  # 错分为wrong_class_num个类
    if len(dic)>0:
        max_class_num = max(dic.values())   # 统计错分到哪类的数量最大
    else :
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

np.save('./featureExtraction/ImageNet_'+str(total_sample_num)+'samplesWhiteFeature.npy',{
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



