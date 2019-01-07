#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VGGの学習済み重みを使わない
"""
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


import pickle
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from data_utils import *
from settings import *


def batch_generator(datapath='traindata_split.dump', batchsize=128, step=128, new_shape=[64,64]):
    with open(datapath, mode='rb') as f:
        data = pickle.load(f)
    image = data['image']
    label = data['label']
    numData = image.shape[0]
    idx = 0
    # print(depthcolor.shape)
    # print(np.max(depth))
    K = np.max(image)
    print(K)


    while True:
        if idx == 0:
            perm1 = np.arange(batchsize * step)
            np.random.shuffle(perm1)
            x,y = np.random.randint(64, size=2)
            image_crop = image[:,y:y+256,x:x+256].astype(np.float32)/255
            label_crop = label[:,y:y+256,x:x+256].astype(np.float32)/255

        batchx = image_crop[perm1[idx:idx + batchsize]]
        batchx = np.transpose(batchx, [0, 3, 1, 2])
        batchy = label_crop[perm1[idx:idx + batchsize]]
        batchy = np.transpose(batchy, [0, 3, 1, 2])
        # print(batchx1.shape)
        # print(batchx2.shape)
        # print(batchy.shape)


        yield batchx, batchy

        if idx + batchsize >= batchsize * step:
            idx = 0
        elif idx + batchsize >= image_crop.shape[0]:
            idx = 0
        else:
            idx += batchsize


# parameters
threshold = 0.1
EPOCH = 10
BATCHSIZE = 8
NUM_DATA = 1747
size = [256,256]
num_batches = int(NUM_DATA / BATCHSIZE)
# load model
loadpath = DATA_DIR + 'weight/fc1e0'
f = open(loadpath+'.json', 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string)
train_model.load_weights(loadpath+'_W.hdf5')



datapath = DATA_DIR + 'patches/traindata.pkl'
gen = batch_generator(datapath=datapath, batchsize=BATCHSIZE, step=num_batches, new_shape=size)


for epoch in range(EPOCH):
    testdata = next(gen)
    y = train_model.predict(testdata[0]) # [8, 5, 256, 256]
    y = np.transpose(y, [0,2,3,1])
    y = np.mean(y, axis=3)
    print(y.shape)

    y = np.minimum(1, y)
    y = np.maximum(0, y)

    image = testdata[0]
    image = np.transpose(image, [0,2,3,1])

    merge = image + y[:,:,:,np.newaxis]
    merge = np.minimum(1, merge)
    merge = np.maximum(0, merge)


    image = (image*255).astype(np.uint8)
    y = (y * 255).astype(np.uint8)
    merge = (merge * 255).astype(np.uint8)

    for i in range(BATCHSIZE):
        plt.subplot(1,3,1)
        plt.imshow(image[i])
        plt.subplot(1,3,2)
        plt.imshow(y[i])
        plt.gray()
        plt.subplot(1,3,3)
        plt.imshow(merge[i])
        plt.show()

    # for i in range(8):
    #     plt.subplot(4,6, 3*i+1)
    #     plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
    #                     labelleft='off')
    #     depthcolor = testdata[0][0][i]
    #     # print(depthcolor.shape)
    #     depthcolor = np.transpose(depthcolor, [1,2,0])
    #     # print(depthcolor.shape, depthcolor.dtype)
    #     plt.imshow(depthcolor)
    #     plt.subplot(4,6,3*i+2)
    #     plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
    #                     labelleft='off')
    #     predict = y3[i]
    #     # print(predict.shape, predict.dtype)
    #     plt.imshow(predict)
    #     plt.subplot(4,6,3*i+3)
    #     plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
    #                     labelleft='off')
    #     groundtruth = testdata[1][i]
    #     label = np.zeros(size, dtype=np.uint8)
    #     for k in range(21):
    #         label[groundtruth[k] == 1] = k + 1
    #     # groundtruth = np.transpose(groundtruth, [1,2,0])
    #     # groundtruth = np.sum(groundtruth, axis=-1)
    #
    #     # print(groundtruth.shape, groundtruth.dtype)
    #
    #     plt.imshow(label)
    #
    # num = number_padded = '{0:04d}'.format(epoch)
    # savepath = 'result/' + num + 'rand_f6e32.png'
    # plt.savefig(savepath, dpi=300)
    # # plt.show()
    # plt.cla