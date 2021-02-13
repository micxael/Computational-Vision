"""
Author:   Jae Park
Course:   CSCI 4270
HW1 Problem 1
File:     p1_checkerboard.py
python p1_checkerboard.py input.jpg output.jpg m n
Purpose:

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys


def square_crop(img, in_img_name):

    height = img.shape[0]
    width = img.shape[1]

    # if img is vertically long
    if height > width:
        top = (height - width)//2
        bottom = top + width

        return_img = img[top:bottom, :, :]
        print('Image {} cropped at ({}, {}) and ({}, {})'
              .format(in_img_name, top, 0, width+top-1, width-1))

    # if img is horizontally long
    elif width > height:
        left = (width - height)//2
        right = left + height

        return_img = img[:, left:right, :]
        print('Image {} cropped at ({}, {}) and ({}, {})'
              .format(in_img_name, 0, left, height-1, right-1))

    else:
        # print("Image is already a square")
        return_img = img

    return return_img


def resize_img(img, m):

    if img.shape == (m, m):
        return_img = img
    else:
        return_img = cv2.resize(img, (m, m))
    print('Resized from ({}, {}, {}) to ({}, {}, {})'
          .format(img.shape[0], img.shape[1], img.shape[2], return_img.shape[0], return_img.shape[1], return_img.shape[2]))
    return return_img


def flip_upside_down(img, m):

    #  | 0 1 | => | 0 2 |
    #  | 2 3 |    | 1 3 |
    return_img = np.transpose(img, (1, 0, 2))

    #  | 0 2 | => | 2 3 |
    #  | 1 3 |    | 0 1 |
    return_img = np.rot90(return_img, k=1)

    return return_img


def invert_colors(img):

    return_img = 255 - img[:, :, :]
    # or simply,
    # return_img = 255 - img

    return return_img


def create_tile(n, img_0_0, img_0_1, img_1_0, img_1_1):
    # make 2 x 2 image
    temp_image_above = np.concatenate((img_0_0, img_0_1), axis=1)
    temp_image_below = np.concatenate((img_1_0, img_1_1), axis=1)

    return_img = np.concatenate((temp_image_above, temp_image_below), axis=0)

    return_img = np.tile(return_img, (n, n, 1))

    return return_img


# MAIN
if __name__ == '__main__':

    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 5:
        print('Arguments not provided correctly.')
        sys.exit()

    in_img_name = sys.argv[1]
    out_img_name = sys.argv[2]
    m = int(sys.argv[3])
    n = int(sys.argv[4])

    in_img = cv2.imread(in_img_name)
    if in_img is None:
        print("Failed to open image", sys.argv[1])
        sys.exit()

    # crop img to a square shape
    in_img = square_crop(in_img, in_img_name)
    # resize the square img to m x m pixels
    in_img_0_0 = resize_img(in_img, m)

    in_img_1_1 = flip_upside_down(in_img_0_0, m)

    in_img_0_1 = invert_colors(in_img_0_0)

    in_img_1_0 = invert_colors(in_img_1_1)

    out_img = create_tile(n, in_img_0_0, in_img_0_1, in_img_1_0, in_img_1_1)

    cv2.imwrite(out_img_name, out_img)
    print('The checkerboard with dimensions {} X {} was output to {}'
          .format(out_img.shape[0], out_img.shape[1], out_img_name))

    sys.exit()
