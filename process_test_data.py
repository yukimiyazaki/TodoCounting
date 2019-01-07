import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join, relpath
import glob, os
from scipy.ndimage.filters import gaussian_filter
import pickle
from settings import *
from data_utils import *

from keras import backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout, TimeDistributed, Input, merge, GaussianNoise, BatchNormalization
from keras.layers import LSTM
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2
from keras.models import model_from_yaml
from keras.utils import np_utils
from keras.initializations import normal, zero
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.vgg16 import VGG16


def patch_split_of(image, patch_size):
    w = image.shape[1]//patch_size
    h = image.shape[0]//patch_size
    patch = np.zeros([w*h, patch_size, patch_size, image.shape[2]], dtype=np.uint8)
    for i in range(w*h):
        x = i%w
        y = i//w
        # print(x,y)
        patch[i] = image[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
    return patch


# テストデータのIDリストを得る
image_folder = DATA_DIR + 'Train_blacked/'
label_folder = DATA_DIR + 'label_images/'
id_path_list  = glob.glob(label_folder+'*.pkl')
id_list = np.zeros(len(id_path_list), dtype=np.int16)
for key, item in enumerate(id_path_list):
    id = int(os.path.basename(item)[:-4])
    id_list[key] = id
id_list = np.sort(id_list)
# for item in id_list:
#     print(item)

# 画像を一枚読み込む
imagepath = image_folder + str(id_list[0]) + '.png'
image_pil = Image.open(imagepath)
image = np.array(image_pil)
# print(image.shape, image.dtype)

# 画像をパッチ化する
patch_size = 256
w = image.shape[1]//patch_size + 1
h = image.shape[0]//patch_size + 1
patches = patch_split_of(image, patch_size)
print(patches.shape, patches.dtype)

# パッチのヒートマップを得る
# load model
loadpath = DATA_DIR + 'weight/fc1e0'
f = open(loadpath+'.json', 'r')
json_string = f.read()
f.close()
model = model_from_json(json_string)
model.load_weights(loadpath+'_W.hdf5')

predict = np.zeros([patches.shape[0], patches.shape[1], patches[2], 5], dtype=np.float32)
batch_size = 4
num_batch = patches//batch_size + 1
for i in range(num_batch):
    if i==num_batch-1:
        batch = patches[i * batch_size:]
    else:
        batch = patches[i*batch_size:(i+1)*batch_size]
    y = model.predict(batch)
    y = np.transpose(y, [0,2,3,1])
    if i == num_batch - 1:
        predict[i*batch_size:] = y
    else:
        predict[i*batch_size:(i+1)*batch_size] = y

# 画像のヒートマップを得る
# ヒートマップを保存
# 統計量をcsv保存
# sum
# histogrum

