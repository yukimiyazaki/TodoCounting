import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join, relpath
import glob, os
from scipy.ndimage.filters import gaussian_filter
import pickle
from settings import *
from sklearn.feature_extraction import image




def blacknize_train_images():
    # TODO rotationの組み込み
    coordspath = DATA_DIR + 'coords.csv'
    train_folder = DATA_DIR + 'Train/'
    dot_folder = DATA_DIR + 'TrainDotted/'
    save_folder = DATA_DIR + 'Train_blacked/'
    # train_folder = 'data/TrainSmall2/Train/'
    # dot_folder = 'data/TrainSmall2/TrainDotted/'
    # save_folder = 'data/TrainSmall2/Train_blacked/'

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # 画像データのリストを読み込む
    train_images_list  = glob.glob(train_folder+'*.jpg')
    black_images_list  = glob.glob(save_folder+'*.png')
    black_id_list = []
    for i in black_images_list:
        black_id_list.append(os.path.basename(i)[:-4])
    print(black_id_list)

    # 各画像ごとに処理
    for trainpath in train_images_list:
        print('processing ', trainpath)
        # 画像番号, 画像, 座標を用意
        id = int(os.path.basename(trainpath)[:-4])
        black_path = save_folder + str(id) + '.png'
        # print(os.path.basename(black_path)[:-4])
        if os.path.basename(black_path)[:-4] in black_id_list:
            # print('it is! ')
            continue
        train_image = np.asarray(Image.open(trainpath))
        train_image.flags.writeable = True
        # print(train_image.flags)
        dot_path = dot_folder + str(id) + '.jpg'
        dot_image = np.asarray(Image.open(dot_path))
        maskimage = np.zeros([dot_image.shape[0], dot_image.shape[0]])
        maskimage = np.sum(dot_image, axis=2)
        print(maskimage.shape)
        train_image[maskimage==0] = 0

        # plt.imshow(train_image)
        # plt.show()

        # 保存
        image = Image.fromarray(train_image)
        image.save(black_path)


if __name__=='__main__': blacknize_train_images()