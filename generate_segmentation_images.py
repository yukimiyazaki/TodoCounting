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
coordspath = DATA_DIR + 'coords.csv'
train_folder =  DATA_DIR + 'Train/'
black_folder = DATA_DIR + 'Train_blacked/'
# save_folder = 'H:/KaggleNOAASeaLions/classified_images/'
save_folder = DATA_DIR + 'label_images/'
print(black_id_list)
# 保存
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

# トドの座標データを読み込む
data = pd.read_csv(coordspath)
coord = np.asarray(data.as_matrix())
print(coord.shape)

# 画像データのリストを読み込む
black_images_list  = glob.glob(black_folder+'*.png')
# print(train_images_list)
seg_images_list  = glob.glob(save_folder+'*.pkl')
seg_id_list = []
for i in seg_images_list:
    seg_id_list.append(os.path.basename(i)[:-4])

# 各画像を処理する
for imagepath in black_images_list:
    id = int(os.path.basename(imagepath)[:-4])
    if id in black_id_list:
        print(id, 'is bad id!')
        continue
    if os.path.basename(imagepath)[:-4] in seg_id_list:
        print('it is! ')
        continue
    print('blacked image id ', id)
    image_pil = Image.open(imagepath)
    image = np.asarray(image_pil)
    coord_of_image = coord[coord[:,0]==id]
    print('number of dot: ', coord_of_image.shape[0])
    label = np.zeros([image.shape[0], image.shape[1], 5], np.bool)

    for i in range(coord_of_image.shape[0]):
        cls = coord_of_image[i,1]
        x = coord_of_image[i,3]
        y = coord_of_image[i,2]
        label[y, x, cls] = 1

    print('number of dot in image', np.sum(label))
    # ブラーをかける
    # for i in range(5):
    #     label[:,:,i] = gaussian_filter(label[:,:,i], sigma=15) # 3->15
    # print('max', np.max(label))
    # label = label/blurred_white
    # label = np.minimum(255, label * 255).astype(np.uint8)

    # 可視化
    # label_image = np.sum(label, axis=2)
    # print(label_image.shape)
    # image = image.astype(np.float64)/255
    # image = np.minimum(1, image + label_image[:,:,np.newaxis].astype(np.float64)/255)
    # image = (image*255).astype(np.uint8)
    # plt.imshow(image)
    # plt.show()



    savepath = save_folder + str(id) + '.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(label, f)