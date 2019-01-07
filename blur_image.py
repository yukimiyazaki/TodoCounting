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
import time

startTime = time.time()

data_folder = DATA_DIR + 'patches_bool/'
data_path_list  = glob.glob(data_folder+'*traindata_reduced.pkl')

# ぼかし方を設定
# todo 各dotのガウスを和算せずに最大値を取る -> peakが消失しない ref openpose
sigma = 15
sample = np.zeros([99,99], dtype=np.float32)
sample[44,44] = 1
sample = gaussian_filter(sample, sigma=sigma)
# plt.imshow(sample)
# plt.gray()
# plt.show()
peak = np.max(sample)
# print(peak)

for path in data_path_list:
    id = int(os.path.basename(path)[:-len('traindata_reduced.pkl')])
    print('processing: ', id)
    with open(path, mode='rb') as f:
            dict = pickle.load(f)
    slice = 1000
    images = dict['image'][:slice]
    labels = dict['label'][:slice]
    labels_blurred = np.zeros([slice,labels.shape[1], labels.shape[2], 5], dtype=np.float32)
    # print('labels shape', labels.shape)
    for i in range(labels.shape[0]):
        print(i)
        label = labels[i].astype(np.float32)
        # print(np.max(label))
        # print(label.shape)
        blurred = np.zeros_like(label, dtype=np.float32)
        blurred = gaussian_filter(label[:, :], sigma=15)
        for ch in range(label.shape[2]):
            blurred[:,:,ch] = gaussian_filter(label[:,:,ch], sigma=sigma)
            # print(np.max(blurred))
        labels_blurred[i] = blurred

    # labels_blurred = labels_blurred/peak/2
    print('label peak ', np.max(labels_blurred))
    labels_blurred = np.minimum(1, labels_blurred)

    # 可視化
    # for i in range(slice):
    #     plt.subplot(2,3,1)
    #     plt.imshow(images[i])
    #     plt.subplot(2,3,2)
    #     plt.imshow(labels_blurred[i,:,:,0])
    #     plt.gray()
    #     plt.subplot(2,3,3)
    #     plt.imshow(labels_blurred[i,:,:,1])
    #     plt.gray()
    #     plt.subplot(2,3,4)
    #     plt.imshow(labels_blurred[i,:,:,2])
    #     plt.gray()
    #     plt.subplot(2,3,5)
    #     plt.imshow(labels_blurred[i,:,:,3])
    #     plt.gray()
    #     plt.subplot(2,3,6)
    #     plt.imshow(labels_blurred[i,:,:,4])
    #     plt.gray()
    #     plt.show()

    # 保存
    dict = {'image': images, 'label': labels_blurred}
    savepath = DATA_DIR + str(id) + '_train_blurred.pkl'
    with open(savepath, mode='wb') as f:
        pickle.dump(dict, f)
    print('saved: ', savepath, time.time()-startTime)