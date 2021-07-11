"""
Author:   Chuck Stewart
Course:   CSCI 4270 and 6270
Lecture:  02
File:     ex3_side_by_side.py

Purpose: Combine two randomly selected images, showing them
side-by-side by creating a single scaled image and using OpenCV's
imshow function.  Save them to the same folder as the original image.  
"""

import cv2
import numpy as np
import os
import random
import sys

"""
Handle the command-line arguments
"""
if len(sys.argv) != 2:
    print("Usage: %s image-folder" % sys.argv[0])
    print("where image-folder is a path to a folder containing images.")
    sys.exit()

test_image_dir = sys.argv[1]
os.chdir(test_image_dir)
img_list = os.listdir('./')
img_list = [name for name in img_list if 'jpg' in name.lower()]

"""
Through random sampling of indices, get two images that are the same size.
"""
random.seed()       # generate a random seed
found = False
while not found:
    ii = random.randint(0, len(img_list) - 1)
    jj = random.randint(0, len(img_list) - 1)
    print("ii = %d, jj = %d" % (ii, jj))
    if ii == jj:
        continue
    img_i = cv2.imread(img_list[ii])
    img_j = cv2.imread(img_list[jj])
    print(img_i.shape)
    print(img_j.shape)
    found = img_i.shape == img_j.shape

"""
Set the width of each half of the output image and select the
height based on the aspect ratio of the images.
"""
out_width = 600
out_height = int(out_width * img_i.shape[0] / img_i.shape[1])

"""
Create resized output images
"""
out1 = cv2.resize(img_i, (out_width, out_height))
out2 = cv2.resize(img_j, (out_width, out_height))

"""
Here are two different methods for creating the output image.  One
creates an NumPy array of the appropriate size and inserts the output
image through slicing.  The other uses concatenation.  The second method
is preferred.
"""
if False:
    out_img = np.zeros((out_height, 2*out_width, 3), dtype=img_i.dtype)
    out_img[:out_height, :out_width] = out1 # cv2.resize(img_i, (out_width, out_height))
    out_img[:out_height, out_width:] = out2 # cv2.resize(img_j, (out_width, out_height))
else:
    out_img = np.concatenate((out1, out2), axis=1)

"""
Now make the display
"""
cv2.imshow('Side-by-side', out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Finally, for the first time we are going to save the image to a file.
Note that in none of this do we need to reverse the order of the
colors in the NumPy arrays.
"""
name_ii = os.path.splitext(img_list[ii])[0]
name_ext = os.path.splitext(img_list[ii])[-1]
name_jj = os.path.splitext(img_list[jj])[0]
out_name = name_ii + '_' + name_jj + name_ext
cv2.imwrite(out_name, out_img)
print("Wrote the result to", out_name)
