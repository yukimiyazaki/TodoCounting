"""
red: adult males
magenta: subadult males
brown: adult females
blue: juveniles
green: pups
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob, os


def get_label(train_image, dot_image):
    color_list = [[255,0,0], [255,0,255], [90, 50, 0], [0,0,255], [0,255,0]]
    train_image = train_image.astype(float)
    dot_image = dot_image.astype(float)

    red_image = np.linalg.norm(dot_image - color_list[0], axis=2) # 2**1
    magenta_image = np.linalg.norm(dot_image - color_list[1], axis=2) # 2**2
    brown_image = np.linalg.norm(dot_image - color_list[2], axis=2) # 2**3
    blue_image = np.linalg.norm(dot_image - color_list[3], axis=2) # 2**4
    green_image = np.linalg.norm(dot_image - color_list[4], axis=2) # 2**5
    black_image = np.linalg.norm(dot_image, axis=2) # 2**6
    diff_image = np.linalg.norm(train_image - dot_image, axis=2) # 2**7

    # trainとdotで差があるところをラベル
    diff_label_image = np.zeros([train_image.shape[0], train_image.shape[1]])
    diff_label_image[diff_image >= 80] += 1

    # 差がありかつ黒いところ (禁止エリア)
    yellow_label_image = np.array(diff_label_image)
    yellow_label_image[black_image <= 30] += 1

    # 差がありかつredいところ
    red_label_image = np.array(diff_label_image)
    red_label_image[red_image <= 120] += 1

    # 差がありかつmagentaいところ
    magenta_label_image = np.array(diff_label_image)
    magenta_label_image[magenta_image <= 120] += 1

    # 差がありかつbrownいところ
    brown_label_image = np.array(diff_label_image)
    brown_label_image[brown_image <= 30] += 1

    # 差がありかつblueいところ
    blue_label_image = np.array(diff_label_image)
    blue_label_image[blue_image <= 140] += 1

    # 差がありかつgreenいところ
    green_label_image = np.array(diff_label_image)
    green_label_image[green_image <= 110] += 1

    # ドットを抽出した画像を作成する
    label_image = np.zeros_like(train_image, dtype=np.uint8)
    label_image[diff_label_image==1] = [255,255,255]
    label_image[red_label_image==2] = color_list[0]
    label_image[magenta_label_image==2] = color_list[1]
    label_image[brown_label_image==2] = color_list[2]
    label_image[yellow_label_image==2] = [255,255,0]
    label_image[blue_label_image==2] = color_list[3]
    label_image[green_label_image==2] = color_list[4]
    # plt.imshow(merged_image)
    # plt.show()

    # ラベル配列作成
    label = np.zeros([train_image.shape[0], train_image.shape[1]], dtype=np.uint8)
    label[red_label_image==2] = 1
    label[magenta_label_image==2] = 2
    label[brown_label_image==2] = 3
    label[yellow_label_image==2] = 6
    label[blue_label_image==2] = 4
    label[green_label_image==2] = 5

    return label, label_image


def process_one_image():
    train_image_path = "minidata/Train/0.jpg"
    train_image = Image.open(train_image_path)
    train_image = np.asarray(train_image)

    dot_image_path = "minidata/TrainDotted/0.jpg"
    dot_image = Image.open(dot_image_path)
    dot_image = np.asarray(dot_image)
    label, _ = get_label(train_image, dot_image)

    patch_size = 100
    for i in range(train_image.shape[0]//patch_size):
        for j in range(train_image.shape[1]//patch_size):
            # print(i, j)
            train_patch = train_image.astype(np.uint8)[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            dot_patch = dot_image.astype(np.uint8)[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            label_patch = label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if len(np.where(label_patch==3)[0]): # 赤の点があるなら
                plt.subplot(1, 3, 1)
                plt.imshow(train_patch)
                plt.subplot(1, 3, 2)
                plt.imshow(dot_patch)
                plt.subplot(1, 3, 3)
                plt.imshow(label_patch)
                plt.show()


def process_images_in_folder(dot_folder='minidata/TrainDotted/',
                             train_folder='minidata/Train/',
                             label_folder='minidata/Label/',
                             label_image_folder='minidata/LabelImage/'):
    files = glob.glob(dot_folder+'*.jpg') # ドット付き画像のパスリストを得る
    for file in files: # 各画像について処理
        dot_image_path = file
        train_image_path = train_folder + os.path.basename(file)
        dot_image_pil, train_image_pil = Image.open(dot_image_path), Image.open(train_image_path)
        dot_image, train_image = np.asarray(dot_image_pil), np.asarray(train_image_pil)
        print(dot_image.shape, train_image.shape)
        label, label_image  = get_label(train_image, dot_image)
        label_pil = Image.fromarray(label)
        label_path = label_folder + os.path.basename(file)[:-4] + '.png'
        label_pil.save(label_path)
        label_image_pil = Image.fromarray(label_image)
        label_image_path = label_image_folder + os.path.basename(file)[:-4] + '.png'
        label_image_pil.save(label_image_path)


        print(label_path, ' saved.')


def check_label_images():
    imagepath = 'minidata/Label/1.png'
    image_pil = Image.open(imagepath)
    image = np.asarray(image_pil)
    print('image shape, type', image.shape, image.dtype)
    plt.imshow(image)
    plt.show()


def main():
    process_images_in_folder(dot_folder='data/TrainSmall2/TrainDotted/',
                             train_folder='data/TrainSmall2/Train/',
                             label_folder='data/TrainSmall2/Label/',
                             label_image_folder='data/TrainSmall2/LabelImage/')

if __name__=='__main__': main()