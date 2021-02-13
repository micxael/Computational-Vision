"""
Author:   Jae Park
Course:   CSCI 4270
Lecture exercise 02
File:     lec02_ex.py

Purpose: The script should read the
image and create two copies of it: I_h should be a resized version with half the number of rows and
columns, and I_q should be a resized version with one-fourth the number of rows and columns. The
output image should show the original image with I_h written over top of it (and centered) and I_q
also written over top of I_h and also centered.
"""

import cv2
import numpy as np

import os
import sys

"""
Handle the command-line arguments
"""
if len(sys.argv) != 3:
    print("Arguments not provided correctly.")
    sys.exit()

in_img_name = sys.argv[1]
out_img_name = sys.argv[2]

input_img = cv2.imread(in_img_name)
if input_img is None:
    print("Failed to open image", sys.argv[1])
    sys.exit()

(row, column) = (input_img.shape[0], input_img.shape[1])

I_h = cv2.resize(input_img, (column // 2, row // 2))
I_q = cv2.resize(input_img, (column // 4, row // 4))

print(I_q.shape[0], I_q.shape[1])

# broadcast two smaller images I_h & I_q onto input_img
input_img[row//4: row//4 + I_h.shape[0], column//4: column//4 + I_h.shape[1], :] = I_h

input_img[row//8 * 3: row//8 * 3 + I_q.shape[0], column//8 * 3: column//8 * 3 + I_q.shape[1], :] = I_q

cv2.imwrite(out_img_name, input_img)
print("Wrote the result to", out_img_name)

sys.exit()
