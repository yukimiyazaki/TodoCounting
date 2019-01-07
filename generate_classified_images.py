import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import glob, os
import pandas as pd

# パスは各環境に合わせて書き換える
coordspath = 'data/coords.csv'
train_folder = 'H:/KaggleNOAASeaLions/Train/'
save_folder = 'H:/KaggleNOAASeaLions/classified_images/'

data = pd.read_csv(coordspath)
print(data)

coord = np.asarray(data.as_matrix())
print(coord.shape)


crop_size = 512
num_opened_image = 'XX'
for i in range(coord.shape[0]):
    num_image = coord[i, 0]
    x = coord[i, 3]
    y = coord[i, 2]
    cls = coord[i,1]
    if num_image != num_opened_image:
        filepath = train_folder + str(coord[i,0]) + '.jpg'
        image_pil = Image.open(filepath)
        image = np.asarray(image_pil)
        width = image.shape[1]
        height = image.shape[0]
    close2edge = (x-crop_size//2 < 0) or (x+crop_size//2 > width) or (y-crop_size//2 < 0) or (y+crop_size//2 > height)
    # print(close2edge)
    if not close2edge:
        # print(image.shape)
        # print(coord[i])
        crop_image = image[y-crop_size//2:y+crop_size//2, x-crop_size//2:x+crop_size//2]
        # plt.imshow(crop_image)
        # plt.show()
        crop_image_pil = Image.fromarray(crop_image)
        savepath = save_folder + str(cls+1) +'/' + str(i) + '.png'
        crop_image_pil.save(savepath)
        print(savepath, ' saved')
    num_opened_image = num_image