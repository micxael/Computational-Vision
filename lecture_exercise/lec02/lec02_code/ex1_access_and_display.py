"""
Author:   Chuck Stewart
Course:   CSCI 4270 and 6270
Lecture:  02
File:     ex1_access_and_display.py

Purpose: Very first example of reading images with OpenCV, accessing
them with NumPy and displaying them with either OpenCV or MatPlotLib.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import os
import random
import sys

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("MatPlotLib version:", matplotlib.__version__)


def get_random_image(dir_name, should_list_images=False):
    """
    Return the name of a randomly-chosen image from the given folder.
    Show the list of images in the folder if that is requested.
    """
    random.seed()
    os.chdir(dir_name)
    img_list = os.listdir('./')
    img_list = [name for name in img_list if 'jpg' in name.lower()]
    if should_list_images:
        print("Here are the images in %s" % dir_name)
        for i_name in img_list:
            print(i_name)
        print()

    ii = random.randint(0, len(img_list) - 1)
    return img_list[ii]


def display_img_opencv_full_size(img_name, img):
    """
    We can display images directly through OpenCV.
    """
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_img_opencv_resize_first(img_name, img, new_max_dim=512):
    """
    We can resize the image before displaying it.  This is good
    for thumbnails.
    """
    max_dim = max(img.shape)
    scale = new_max_dim / max_dim
    new_height = int(img.shape[0] * scale)
    new_width = int(img.shape[1] * scale)
    img_1 = cv2.resize(img, (new_width, new_height))
    name_to_display = 'Resized ' + img_name
    cv2.imshow(name_to_display, img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_img_matplotlib(img_name, img):
    """
    Use the PyPlot function from MatPlotLib to show
    the original image.  Scaling occurs internally.
    """
    plt.axis("off")
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 3:
        print("Usage: %s image-or-folder  display-option\n" % sys.argv[0])
        print("where image-or-folder is an image file or a path to a folder")
        print("containing images and display-option is an integer in the")
        print("range 1-4.")
        sys.exit()
    elif os.path.isdir(sys.argv[1]):
        test_image_dir = sys.argv[1]
        img_name = get_random_image(test_image_dir, should_list_images=True)
        if not img_name:
            print("Failed to find images in", test_image_dir)
            sys.exit()
    else:
        img_name = sys.argv[1]

    """
    Open the image using OpenCV
    """
    img = cv2.imread(img_name)
    print('Image name:', img_name)
    print('Image type:', type(img))  # Shows that this is a NumPy array
    print('Image shape:', img.shape)

    """
    Accessing values can occur through NumPy indexing, or through the
    NumPy array method item and itemset.  The latter two are more
    efficient, but we will not use either very often because we will
    use functions and operators applied to entire arrays.
    """
    (ctr_row, ctr_col) = (img.shape[0] // 2, img.shape[1] // 2)
    print('Center pixel value (through indexing):', img[ctr_row, ctr_col])
    print('Upper left corner blue value (through item method): ',
          img.item(0, 0, 0)) #BGR
    print('Upper left corner blue value (through indexing): ', img[0, 0, 0]) #BGR

    """
    Converting to gray scale
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print()
    print('Shape after conversion is', img_gray.shape)
    print('Center pixel value is now', img_gray[ctr_row, ctr_col])

    """
    Examine the result of different display possibilities.  For now
    we'll just show the RGB image.
    """
    display_option = int(sys.argv[2])
    if display_option == 1:
        # Use the default through OpenCV
        display_img_opencv_full_size(img_name, img)

    elif display_option == 2:
        # Resize the image before displaying it
        display_img_opencv_resize_first(img_name, img)

    elif display_option == 3:
        # Use Matplotlib's pyplot options
        display_img_matplotlib(img_name, img)

    elif display_option == 4:
        # Reverse the order of the last axis of the 3d matrix
        # representing the image so that BGR becomes RGB
        # i.e. the colors become inverted
        img_2 = img[:, :, ::-1]  # shallow copy
        display_img_matplotlib(img_name, img_2)
