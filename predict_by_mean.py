"""
画像サイズで線形回帰する
"""
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import glob, os
import pandas as pd

coordspath = 'data/coords.csv'
data = pd.read_csv(coordspath)
coord = np.asarray(data.as_matrix())

y_path = 'H:/KaggleNOAASeaLions/Train/train.csv'
y_data = pd.read_csv(y_path)
y = np.asarray(y_data.as_matrix())
y_count = y[:,1:]
y_avr = np.mean(y_count, axis=0)

print(y_avr)

train_nb = 947
test_nb = 18636

pred = np.zeros([test_nb, 5])
pred[:,:] = y_avr
pred = pred.astype(int) - 1
print(pred[500:505])

pred_data = pd.DataFrame(pred)
# pred_data.columns=['train_id', 'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
print(pred_data)
pred_data.to_csv( 'data/pred_by_mean_minus_one.csv', header=False )


test_folder = 'H:/KaggleNOAASeaLions/Train/'
train_folder = 'H:/KaggleNOAASeaLions/Train/'


image_size_list = np.zeros(train_nb, dtype=int)
# image size = (5616, 3744) or (4992, 3328)
#
# for i in range(train_nb):
#     filepath = train_folder + str(i) + '.jpg'
#     image_pil = Image.open(filepath)
#     width = image_pil.size[0]
#     image_size_list[i] = (width==4992)*1 + (width==5616)*2
#     print(filepath, image_pil.size, image_size_list[i])