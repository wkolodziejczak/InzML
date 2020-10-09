import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from sklearn import tree

#Load images to array
path = "dataset/"
img_list = os.listdir(path)
img_array = []
print(img_list)
for img in img_list:
    img_array.append(cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE))

#Create plot
fig_height = 2
fig_width = 5
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


# # Hardcoded NumPy
# # Open Images
# img0 = Image.open("dataset/0.png").convert('L')
# img1 = Image.open("dataset/1.png").convert('L')
# img2 = Image.open("dataset/2.png").convert('L')
# img3 = Image.open("dataset/3.png").convert('L')
# img4 = Image.open("dataset/4.png").convert('L')
# img5 = Image.open("dataset/5.png").convert('L')

# #Put them in NumPy Array
# arr0 = np.array(img0)
# arr1 = np.array(img1)
# arr2 = np.array(img2)
# arr3 = np.array(img3)
# arr4 = np.array(img4)
# arr5 = np.array(img5)

# # Put them in a plot
# fig1 = plt.figure()
# ax0 = plt.subplot2grid((2,3), (0,0))
# ax1 = plt.subplot2grid((2,3), (0,1))
# ax2 = plt.subplot2grid((2,3), (0,2))
# ax3 = plt.subplot2grid((2,3), (1,0))
# ax4 = plt.subplot2grid((2,3), (1,1))
# ax5 = plt.subplot2grid((2,3), (1,2))

# # Display image in Grayscale
# ax0.imshow(arr0, cmap='gray')
# ax1.imshow(arr1, cmap='gray')
# ax2.imshow(arr2, cmap='gray')
# ax3.imshow(arr3, cmap='gray')
# ax4.imshow(arr4, cmap='gray')
# ax5.imshow(arr5, cmap='gray')

# # Print arrays containing pixel values
# print(arr0)
# print(arr1)
# print(arr2)
# print(arr3)
# print(arr4)
# print(arr5)

# # Show plot
# plt.show()