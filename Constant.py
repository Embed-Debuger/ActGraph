import os
import tensorflow as tf
import numpy as np
import random
from keras import backend as K

SEED = 0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(session)


if __name__ == "__main__":
    pass

