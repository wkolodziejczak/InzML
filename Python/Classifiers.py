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
img_array = []

for img in img_list:
    # Load images into an Array
    img_array.append(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE))

    # Parse file names
    file_name, extension = os.path.splitext(img)
    font_number, number_value = file_name.split(".")
    img_data_list = [font_number, number_value]


# Create plot
fig = plt.figure()
for i in range(fig_height):

    for j in range(fig_width):
        plt.subplot2grid((fig_height, fig_width), (i, j))
        plt.imshow(img_array[j+offset], cmap="gray")
    offset += fig_width

print(len(img_array))
print(img_list[199])
print(img_array[199])
# plt.show()
