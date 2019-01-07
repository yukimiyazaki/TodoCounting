import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import glob, os

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray


def count(image):
    """
    1chの二次元配列のラベルから数を数える
    labelから特定のラベルのみを抜き出したもの
    :param image: nparray [縦, 横]
    :return: int 数えあげた数
    """

def blob_detection(image):
    print(type(image))
    image_gray = rgb2gray(image)
    print(type(image_gray), image_gray.dtype, image_gray.shape)
    print(image_gray[0,0])
    image_gray[0,0] +=0.01

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    print(blobs_log.shape==(0,))

    # Compute radii in the 3rd column.
    if not blobs_log.shape==(0,):
        print(blobs_log)
        print(blobs_log==[])
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    if not blobs_dog.shape==(0,):
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()


def blob_detection2(image):
    image[0,0] +=0.01 # 全部0だとエラーになるので回避

    blobs_dog = blob_dog(image, max_sigma=10, threshold=.2)
    if not blobs_dog.shape==(0,): # ブロブが検出されているなら半径を計算
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    # 図を表示
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    print('数は', blobs_dog.shape[0])
    # 中心(0.2,0.2)で半径0.2の円を描画
    for blob in blobs_dog:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
    plt.show()


def main():
    imagepath = 'data/TrainSmall2/Label/41.png'
    image_pil = Image.open(imagepath)
    x, y = 1850, 2600
    # x, y = 0, 0
    image = np.asarray(image_pil)#[x:x+500, y:y+500]
    # print(image.dtype)
    label_red = np.zeros_like(image, dtype=np.uint8)
    label_red[image==3] = 255
    # print(np.sum(label_red))
    # plt.imshow(label_red)
    # plt.gray()
    # plt.show()

    # TODO ぼかす
    label_red_pil = Image.fromarray(label_red)
    label_red_pil_blur = label_red_pil.filter(ImageFilter.GaussianBlur(radius=5))
    label_red_blur = np.asarray(label_red_pil_blur).astype(np.float64)

    # 画像確認
    # plt.subplot(1,3,1)
    # plt.imshow(image)
    # plt.subplot(1,3,2)
    # plt.imshow(label_red)
    # plt.subplot(1,3,3)
    # plt.imshow(label_red_blur)
    # plt.show()

    # TODO ブロブを数える
    blob_detection2(label_red_blur)

if __name__=='__main__': main()