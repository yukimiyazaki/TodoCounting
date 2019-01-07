import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

color_list = [[255,0,0], [255,0,255], [90, 50, 0], [0,0,255], [0,255,0]]

def prelabel1(train_pixel, dot_pixel, th_black=5, th_diff=50, th_col=50):
    """
    if r > 204 and g < 26 and b < 29: # RED
    elif r > 220 and g > 204 and b < 25: # MAGENTA
    elif 6 < r < 64 and g < 52 and 156 < b < 199: # GREEN
    elif r < 78 and  124 < g < 221 and 31 < b < 85: # BLUE
    elif 59 < r < 115 and g < 49 and 19 < b < 80:  # BROWN
    :param train_pixel:
    :param dot_pixel:
    :param th_black:
    :param th_diff:
    :param th_col:
    :return:
    """
    train_pixel = train_pixel.astype(np.float64)
    dot_pixel = dot_pixel.astype(np.float64)
    black = (np.linalg.norm(dot_pixel) <= th_black)
    different = (np.linalg.norm(dot_pixel - train_pixel) >= th_diff)
    r = dot_pixel[0]
    g = dot_pixel[1]
    b = dot_pixel[2]
    color = 0
    merged_pixel = [0,0,0]
    if r > 204 and g < 26 and b < 29: color = 1
    elif r > 220 and g > 204 and b < 25: color = 2
    elif 6 < r < 64 and g < 52 and 156 < b < 199: color = 5
    elif r < 78 and  124 < g < 221 and 31 < b < 85: color = 4
    elif 59 < r < 115 and g < 49 and 19 < b < 80: color = 5
    if black and different:
        merged_pixel = [255, 255, 0]
    if (not black) and different:
        merged_pixel = [255, 255, 255]
    if (not black) and different and color:
        merged_pixel = color_list[color - 1]

    return merged_pixel

def labeled_image_made_from(train_image, dot_image, th_black=30, th_diff=50, th_col=50):
    merged_image = np.array(dot_image)
    for i in range(train_image.shape[0]):
        for j in range(train_image.shape[1]):
            merged_image[i,j] = prelabel1(train_image[i,j], dot_image[i,j], th_black, th_diff, th_col)
    return merged_image
"""
red: adult males
magenta: subadult males
brown: adult females
blue: juveniles
green: pups

"""


train_image_path = "minidata/Train/0.jpg"
train_image = Image.open(train_image_path)
train_image = np.asarray(train_image)

dot_image_path = "minidata/TrainDotted/0.jpg"
dot_image = Image.open(dot_image_path)
dot_image = np.asarray(dot_image)

patch_size = 100
# train_patch = train_image[:patch_size, :patch_size]
# dot_patch = dot_image[:patch_size, :patch_size]
# merged_patch = labeled_image_made_from(train_patch, dot_patch)
#
# plt.subplot(1, 3, 1)
# plt.imshow(train_patch)
# plt.subplot(1, 3, 2)
# plt.imshow(dot_patch)
# plt.subplot(1, 3, 3)
# plt.imshow(merged_patch)
# plt.show()
#
#
#
#
for i in range(train_image.shape[0]//patch_size):
    for j in range(train_image.shape[1]//patch_size):
        # print(i, j)
        train_patch = train_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
        dot_patch = dot_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
        merged_patch = labeled_image_made_from(train_patch, dot_patch)
        plt.subplot(1, 3, 1)
        plt.imshow(train_patch)
        plt.subplot(1, 3, 2)
        plt.imshow(dot_patch)
        plt.subplot(1, 3, 3)
        plt.imshow(merged_patch)
        plt.show()









# merged_image = np.ones_like(dot_image)
# merged_image[subtracted_image==0] = 0
# plt.imshow(subtracted_image)
# plt.show()
