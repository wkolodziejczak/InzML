import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import tree

# Variables
fig_height = 5
fig_width = 12
offset = 0
path = "dataset/"
img_list = os.listdir(path)

img_array = np.array([])
for img in img_list:
    # Load images into an Array
   np.append(img_array, cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE))

print(img_array.shape)

fig = plt.figure()
for i in range(fig_height):

    for j in range(fig_width):
        plt.subplot2grid((fig_height, fig_width), (i, j))
        plt.imshow(img_array[j+offset], cmap="gray")
    offset += fig_width
