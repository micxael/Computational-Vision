"""
Author:   Chuck Stewart
Course:   CSCI 4270 and 6270
Lecture:  02
File:     ex2_combine_images.py

Purpose:  Demonstrate linearly combining two images using both OpenCV
functions and NumPy functions.  Also demonstrate output of multiple
images in a single MatPlotLib figure.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import sys

"""
Handle the command-line arguments
"""
if len(sys.argv) != 3:
    print("Usage: %s image-folder blend-wgt" % sys.argv[0])
    print("where image-folder is a path to a folder containing images, and")
    print("blend-wgt is the blending weight between images in [0,0.5].")
    sys.exit()

"""
The user provides the combination weight through the command line.
"""
blend_wgt = float(sys.argv[2])
if blend_wgt < 0 or blend_wgt > 0.5:
    print("Blending weight must be in range [0,0.5]")
    sys.exit()

"""
Through random sampling of indices, get two images that are the same size.
"""
test_image_dir = sys.argv[1]
os.chdir(test_image_dir)
img_list = os.listdir('./')
img_list = [name for name in img_list if 'jpg' in name.lower()]
random.seed()
found = False
while not found:
    ii = random.randint(0, len(img_list) - 1)
    jj = random.randint(0, len(img_list) - 1)
    if ii == jj:
        continue
    print("ii = %d, jj = %d" % (ii, jj))
    img_i = cv2.imread(img_list[ii])
    img_j = cv2.imread(img_list[jj])
    print(img_i.shape)
    print(img_j.shape)
    found = img_i.shape == img_j.shape

"""
Weighted linear combination using NumPy operations
"""
img_np = (1-blend_wgt)*img_i + blend_wgt*img_j
print('\nThe data type after use of NumPy operations:', img_np.dtype)
img_np = img_np.astype(img_i.dtype)    # convert back to uint8
print('Converting back to the original type:', img_np.dtype)

"""
Weight linear combination using OpenCV operations.
"""
img_cv = cv2.addWeighted(img_i, blend_wgt, img_j, 1-blend_wgt, 0)
print('\nThe data type after use of OpenCV function:', img_cv.dtype)

"""
Reverse the images (BGR -> RGB) for PyPlot display
"""
img_i = img_i[:, :, ::-1]
img_j = img_j[:, :, ::-1]
img_np = img_np[:, :, ::-1]
img_cv = img_cv[:, :, ::-1]

"""
The display shows a 2x2 grid of images.  Therefore, the number xyz
(e.g. 223) provided to subplot means x rows, y columns, and the z-th
plot in row-major order.
"""
plt.subplot(221)
plt.axis("off")
plt.imshow(img_i)

plt.subplot(222)
plt.axis("off")
plt.imshow(img_j)

plt.subplot(223)
plt.axis("off")
plt.imshow(img_np)

plt.subplot(224)
plt.axis("off")
plt.imshow(img_cv)

plt.show()   # Images do not appear until this point in the code
