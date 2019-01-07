#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
based on fcn semantic segmentation of Long
and
body part segmentation
1222sec/epoch
"""
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout, TimeDistributed, Input, merge, GaussianNoise, BatchNormalization
from keras.layers import LSTM
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2
from keras.models import model_from_yaml
from keras.utils import np_utils
# from keras.initializations import normal, zero
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.applications.vgg16 import VGG16


import pickle
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from settings import *
from data_utils import *
from keras.models import Sequential, Model, model_from_json

startTime = time.time()


def fcn_model(colorshape=[3, 384, 512]):
    cls = 5
    color_input = Input(shape=colorshape)

    x = GaussianNoise(0.1)(color_input)
    # use VGG
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(mp1)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    mp2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(mp2)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    mp3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(mp3)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    mp4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(mp4)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    mp5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # reduce channel
    x = Convolution2D(4096, 7, 7, activation='relu', border_mode='same', name='fc6')(mp5) # => [?, 4096, 12, 16]
    x = Dropout(0.5)(x)
    x = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='fc7')(x) # => [?, 4096, 12, 16]
    x = Dropout(0.5)(x)

    # Deconv Layer
    x = Convolution2D(cls, 4, 4, activation='relu', border_mode='same')(x)  # => [?, 21, 12, 16]
    x = UpSampling2D()(x) # => [?, 21, 24, 16]
    x = merge([x, mp4], mode='concat', concat_axis=-3)
    x = Convolution2D(cls, 4, 4, activation='relu', border_mode='same')(x)  # => [?, 21, 12, 16]
    x = UpSampling2D()(x) # => [?, 21, 24, 16]
    x = merge([x, mp3], mode='concat', concat_axis=-3)
    x = Convolution2D(cls, 4, 4, activation='relu', border_mode='same')(x)  # => [?, 21, 12, 16]
    x = UpSampling2D()(x) # => [?, 21, 24, 16]
    x = merge([x, mp2], mode='concat', concat_axis=-3)
    x = Convolution2D(cls, 4, 4, activation='relu', border_mode='same')(x)  # => [?, 21, 12, 16]
    x = UpSampling2D()(x) # => [?, 21, 24, 16]
    x = merge([x, mp1], mode='concat', concat_axis=-3)
    x = Convolution2D(cls, 4, 4, activation='relu', border_mode='same')(x)  # => [?, 21, 12, 16]
    x = UpSampling2D()(x) # => [?, 21, 24, 16]
    x = merge([x, color_input], mode='concat', concat_axis=-3)
    y = Convolution2D(cls, 4, 4, activation='sigmoid', border_mode='same')(x)  # => [?, 21, 12, 16]
    model = Model(input=color_input, output=y)

    return model


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
    image = np.transpose(image, [0, 3, 1, 2])
    label = np.transpose(label, [0, 3, 1, 2]) # [id, ch, h, w]


    while True:
        if idx == 0:
            perm1 = np.arange(batchsize * step)
            np.random.shuffle(perm1)
            image_epoch = image
            label_epoch = label
            flip_ud = np.random.randint(2)
            if flip_ud:
                image_epoch = image_epoch[:,:, ::-1, :]
                label_epoch = label_epoch[:,:, ::-1, :]
            flip_lr = np.random.randint(2)
            if flip_lr:
                image_epoch = image_epoch[:,:, :, ::-1]
                label_epoch = label_epoch[:,:, :, ::-1]
            x, y = np.random.randint(64, size=2)
            image_epoch = image_epoch[:,:,y:y+256,x:x+256].astype(np.float32)/255
            label_epoch = label_epoch[:,:,y:y+256,x:x+256].astype(np.float32)/255

        batchx = image_epoch[perm1[idx:idx + batchsize]]
        batchy = label_epoch[perm1[idx:idx + batchsize]]
        # print(batchx1.shape)
        # print(batchx2.shape)
        # print(batchy.shape)


        yield batchx, batchy

        if idx + batchsize >= batchsize * step:
            idx = 0
        elif idx + batchsize >= image.shape[0]:
            idx = 0
        else:
            idx += batchsize


# def train():
# parameters
BATCHSIZE = 1
NUM_DATA = 1
EPOCH = 1
size = [256,256]
num_batches = int(NUM_DATA / BATCHSIZE)
datapath = DATA_DIR + 'patches/traindata.pkl'
gen = batch_generator(datapath=datapath, batchsize=BATCHSIZE, step=num_batches, new_shape=size)

# build model
loadpath = DATA_DIR + 'weight/fc1e0'
f = open(loadpath+'.json', 'r')
json_string = f.read()
f.close()
fcn = model_from_json(json_string)
fcn.load_weights(loadpath+'_W.hdf5')
f_opt = Adam(lr=1e-6, beta_1=0.99)
# f_opt = Adam()
fcn.compile(loss='binary_crossentropy', optimizer=f_opt)

# train
# next(gen)
history = fcn.fit_generator(gen, samples_per_epoch=BATCHSIZE * num_batches, nb_epoch=EPOCH)

# save
savepath = DATA_DIR + 'weight/fc1e144'
json_string = fcn.to_json()
with open(savepath+'.json', "w") as f:
    f.write(json_string)
fcn.save_weights(savepath+'_W.hdf5')

model = fcn
new_model = Model(input=model.inputs, output=merged)
funcType = type(model.save)
# monkeypatch the save to save just the underlying model
def new_save(self_,filepath, overwrite=True):
    model.save(filepath, overwrite)
    new_model.save=funcType(new_save, new_model)
    return new_model