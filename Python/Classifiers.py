import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import tree

#Load images to array
path = "dataset/"
img_list = os.listdir(path)
img_array = []
print(img_list)
for img in img_list:
    img_array.append(cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE))

fig_height = 2
fig_width = 4

#Create plot

fig=plt.figure()
for i in range(fig_height):
    if i==0:
        for j in range(fig_width):
            plt.subplot2grid((fig_height,fig_width), (i, j))
            plt.imshow(img_array[j], cmap = "gray")
    else:
        for j in range(fig_width):
            plt.subplot2grid((fig_height,fig_width), (i, j))
            plt.imshow(img_array[j+fig_width], cmap="gray")

plt.show()

