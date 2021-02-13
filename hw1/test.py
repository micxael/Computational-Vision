
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys



"""
for i in range(0,10, 2):
    for j in range(0,10, 2):
        print('i=',i ,'j=',j)
        this = np.average(img[i:i+2,j:j+2])
        print(img[i:i+2,j:j+2])
"""
"""
for i in range(0, 11, 2):
    for j in range(0,10, 2):
        print('i=',i ,'j=',j)
        this = np.average(img[i:i+2,j:j+2])
        print(img[i:i+2,j:j+2])
        print(this)
"""


"""
# List of 4 5x20 image slices
sliced = np.split(img,5,axis=0)
# List of 4 lists of 4 5x5 image blocks
blocks = [np.split(img_slice,5,axis=1) for img_slice in sliced]

"""
"""
arr = np.array(blocks)
# arr = arr.reshape(25, -1, 2,2)
new = np.average(arr, axis=0)
print(arr.shape)
print("----------------")
print(new)
"""

"""
upper_half = np.hsplit(np.vsplit(img, 2)[0], 2)
lower_half = np.hsplit(np.vsplit(img, 2)[1], 2)



print(blocks[0][1])

print(blocks[1])


upper_left = upper_half[0]
upper_right = upper_half[1]
lower_left = lower_half[0]
lower_right = lower_half[1]

print(upper_left)
print(upper_right)
print(lower_left)
"""
#print(blocks[0][1])

#print(blocks[1][1])

#this = np.average(blocks[1][1])
#print(this)