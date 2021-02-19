"""
Author:   Jae Park
Course:   CSCI 4270
HW2 Problem 3
File:     p3_best_focus.py
python p3_best_focus.py /dir_name
Purpose:

"""

import cv2
import numpy as np
import os
import sys


def calc_engy(file):
    filepath = os.path.join(sys.argv[1], file)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    im_dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    im_dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    sum = np.sum(np.square(im_dx)) + np.sum(np.square(im_dy))

    return sum / (img.shape[0] * img.shape[1])


if __name__ == '__main__':
    files = [file for file in os.listdir(sys.argv[1])]
    files.sort()

    max_engy_val = 0
    max_val_file = ''

    for filename in files:
        val = calc_engy(filename)
        print('%s: %.1f' % (filename, val))

        if val > max_engy_val:
            max_engy_val = val
            max_val_file = filename

    print('Image %s is best focused.' % max_val_file)

    sys.exit()
