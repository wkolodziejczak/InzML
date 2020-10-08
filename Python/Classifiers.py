import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from sklearn import tree

# path = "dataset/"
# img_array = []
# print(os.listdir(path))
# for img in os.listdir(path):
#     img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
#     #plt.imshow(img_array,cmap="gray")
#     #plt.show()
# print(img_array)

img0 = Image.open("dataset/0.png").convert('L')
img1 = Image.open("dataset/1.png").convert('L')
img2 = Image.open("dataset/2.png").convert('L')
img3 = Image.open("dataset/3.png").convert('L')
img4 = Image.open("dataset/4.png").convert('L')
img5 = Image.open("dataset/5.png").convert('L')

arr0 = np.array(img0)
arr1 = np.array(img1)
arr2 = np.array(img2)
arr3 = np.array(img3)
arr4 = np.array(img4)
arr5 = np.array(img5)

fig1 = plt.figure()
ax0 = plt.subplot2grid((2,3), (0,0))
ax1 = plt.subplot2grid((2,3), (0,1))
ax2 = plt.subplot2grid((2,3), (0,2))
ax3 = plt.subplot2grid((2,3), (1,0))
ax4 = plt.subplot2grid((2,3), (1,1))
ax5 = plt.subplot2grid((2,3), (1,2))

ax0.imshow(arr0, cmap='gray')
ax1.imshow(arr1, cmap='gray')
ax2.imshow(arr2, cmap='gray')
ax3.imshow(arr3, cmap='gray')
ax4.imshow(arr4, cmap='gray')
ax5.imshow(arr5, cmap='gray')
print(arr0)
print(arr1)
print(arr2)
print(arr3)
print(arr4)
print(arr5)

plt.show()

# print(arr)
# print(arr.shape)
# plt.imshow(img, cmap='gray')
# plt.show()