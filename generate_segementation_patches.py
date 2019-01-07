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
from sklearn.feature_extraction import image

# todo patchがどの画像由来か明示して保存する

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

def label_patch_split_of(image, patch_size):
    w = image.shape[1]//patch_size
    h = image.shape[0]//patch_size
    patch = np.zeros([w*h, patch_size, patch_size], dtype=np.uint8)
    for i in range(w*h):
        x = i%w
        y = i//w
        # print(x,y)
        patch[i] = image[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
    return patch

def generate_patches_from(image, label):
    patch_size = 320
    width = image.shape[1]
    height = image.shape[0]
    new_width = (width//patch_size + 1) * patch_size
    new_height = (height//patch_size + 1) * patch_size

    new_image = np.zeros([new_height, new_width, 3], dtype=np.uint8)
    new_image[:height, :width] = image

    new_label = np.zeros([new_height, new_width, 5], dtype=np.bool)
    new_label[:height, :width] = label

    image_patches = patch_split_of(new_image, patch_size)
    label_patches = patch_split_of(new_label, patch_size).astype(np.bool)
    # print(np.max(label_patches), np.min(label_patches))

    return image_patches, label_patches


def generate_segmentation_patches():
    # パスは各環境に合わせて書き換える
    # train_folder = DATA_DIR + 'Train_blacked/'
    # train_folder = 'data/TrainSmall2/Train_blacked/'
    black_folder = DATA_DIR + 'Train_blacked/'
    label_folder = DATA_DIR + 'label_images/'
    # save_folder = 'H:/KaggleNOAASeaLions/classified_images/'
    save_folder = DATA_DIR + 'patches_bool/'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    patch_list  = glob.glob(save_folder+'*_traindata.pkl')
    patch_id_list = []
    print(patch_list)
    for i in patch_list:
        patch_id_list.append(int(os.path.basename(i)[:-14]))
    print(patch_id_list)

    # ラベルデータのリストを読み込む
    label_path_list  = glob.glob(label_folder+'*.pkl')
    print(label_path_list)

    # 各画像ごとに処理
    for item in label_path_list:
        # 画像とラベルを読み込む
        id = int(os.path.basename(item)[:-4])
        print('processing id: ', id)
        if id in black_id_list: continue
        if id in patch_id_list:
            'already converted'
            continue
        labelpath = label_folder + str(id) + '.pkl'
        with open(labelpath, mode='rb') as f:
            label = pickle.load(f)
        image = np.asarray(Image.open(black_folder + str(id) + '.png'))
        image = image[:,:,:3]
        # plt.imshow(image)
        # plt.show()
        print('shape', image.shape)

        # パッチを生成する
        image_patch, label_patch = generate_patches_from(image, label)
        print('number of patches: ', image_patch.shape[0])

        # for i in range(image_patch.shape[0]):
        #     plt.subplot(1,2,1)
        #     plt.imshow(image_patch[i])
        #     plt.subplot(1,2,2)
        #     label_image = label_patch[i]
        #     label_image = np.mean(label_image, axis=2)
        #     plt.imshow(label_image)
        #     plt.show()

        # 保存
        # print(image_patch.dtype, label_patch.dtype)
        dict = {'image': image_patch, 'label': label_patch}
        savepath = save_folder + str(id) + '_traindata.pkl'
        with open(savepath, mode='wb') as f:
            pickle.dump(dict, f)
        print('saved: ', savepath)


if __name__=='__main__': generate_segmentation_patches()