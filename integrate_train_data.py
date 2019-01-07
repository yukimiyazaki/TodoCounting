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


# パスは各環境に合わせて書き換える
# TODO データのパスを指定するファイルを設ける
coordspath = 'data/coords.csv'
# train_folder = 'H:/KaggleNOAASeaLions/Train/'
data_folder = DATA_DIR + 'patches_bool/'
# save_folder = 'H:/KaggleNOAASeaLions/classified_images/'
save_folder = DATA_DIR + 'patches_bool/'
# 保存
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)


data_path_list  = glob.glob(data_folder+'*_traindata.pkl')
print(data_path_list)


images = np.zeros([0, 320, 320, 3], dtype=np.uint8)
labels = np.zeros([0, 320, 320, 5], dtype=np.bool)
# labels = np.zeros([0, 320, 320], dtype=np.uint8)
data_split = 0
for item in data_path_list:
    print('load: ', os.path.basename(item))
    with open(item, mode='rb') as f:
        dict = pickle.load(f)
    image = dict['image']
    label = dict['label']

    image_sum = np.sum(image, axis=1)
    image_sum = np.sum(image_sum, axis=1)
    image_sum = np.sum(image_sum, axis=1)
    black_reduced_image = np.array(image[image_sum != 0])
    black_reduced_label = np.array(label[image_sum != 0])
    print(black_reduced_image.shape)

    label_sum = np.sum(black_reduced_label, axis=1)
    label_sum = np.sum(label_sum, axis=1)
    label_sum = np.sum(label_sum, axis=1)
    no_label_reduced_image = np.array(black_reduced_image[label_sum != 0])
    no_label_reduced_label = np.array(black_reduced_label[label_sum != 0])
    print('reduced ', image.shape[0], '->', black_reduced_image.shape[0], '->', no_label_reduced_image.shape[0])




    images = np.r_[images, no_label_reduced_image]
    labels = np.r_[labels, no_label_reduced_label]
    print('number of data ', images.shape[0])

    if images.shape[0] >= 1000:
        dict = {'image': images[:1000], 'label': labels[:1000]}
        images = images[1000:]
        labels = labels[1000:]
        savepath = save_folder + str(data_split) + 'traindata_reduced.pkl'
        with open(savepath, mode='wb') as f:
            pickle.dump(dict, f)
        print('saved: ', savepath)
        data_split += 1

# 保存
data_split += 1
dict = {'image': images, 'label': labels}
savepath = save_folder + str(data_split) + 'traindata_reduced.pkl'
with open(savepath, mode='wb') as f:
    pickle.dump(dict, f)
print('saved: ', savepath)