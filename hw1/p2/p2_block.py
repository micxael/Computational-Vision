"""
Author:   Jae Park
Course:   CSCI 4270
HW1 Problem 2
File:     p2_block.py
python p2_block.py input.jpg m n b
Purpose:

"""

import cv2
import numpy as np

import sys


def downsized_img(img, m, n):
    M = img.shape[0]
    N = img.shape[1]
    s_m = M / m
    s_n = N / n

    i_index = []
    for i in range(0, m + 1):
        i_index.append(round(s_m * i))

    j_index = []
    for j in range(0, n + 1):
        j_index.append(round(s_n * j))

    downsized_img = np.zeros((m, n))

    for i in range(0, m):
        for j in range(0, n):
            downsized_img[i, j] = \
                np.average(img[i_index[i]:i_index[i + 1], j_index[j]:j_index[j + 1]])

    print('Downsized images are ({}, {})'.format(m, n))
    print('Block images are ({}, {})'
          .format(downsized_img.shape[0] * b, downsized_img.shape[1] * b))
    print('Average intensity at ({}, {}) is {}'
          .format(m//4, n//4, format(round(downsized_img[m//4, n//4], 2), ".2f")))
    print('Average intensity at ({}, {}) is {}'
          .format(m // 4, (3*n) // 4, format(round(downsized_img[m // 4, (3*n) // 4], 2), ".2f")))
    print('Average intensity at ({}, {}) is {}'
          .format((3*m) // 4, n // 4, format(round(downsized_img[(3*m) // 4, n // 4], 2), ".2f")))
    print('Average intensity at ({}, {}) is {}'
          .format((3*m) // 4, (3*n) // 4, format(round(downsized_img[(3*m) // 4, (3*n) // 4], 2), ".2f")))

    return downsized_img


def binary_img(img):
    return_img = img.copy()

    median_val = np.median(return_img)

    print("Binary threshold:", format(round(median_val, 2), ".2f"))

    return_img = np.where(return_img < median_val, 0, 255)

    return return_img


def upsample_to_block(img, m, n, b):

    img = np.round(img)

    new_img = np.zeros((m * b, n * b))

    for i in range(0, m):
        for j in range(0, n):
            new_img[i * b:(i + 1) * b, j * b:(j + 1) * b] = img[i:i+1, j:j+1]

    return new_img


# MAIN
if __name__ == '__main__':

    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 5:
        print('Arguments not provided correctly.')
        sys.exit()

    input_img_name = sys.argv[1]
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    b = int(sys.argv[4])

    input_img = cv2.imread(input_img_name, cv2.IMREAD_GRAYSCALE)

    img_g = downsized_img(input_img, m, n)

    img_b = binary_img(img_g)

    img_g = upsample_to_block(img_g, m, n, b)
    img_b = upsample_to_block(img_b, m, n, b)

    file_name_list = input_img_name.split(".", 2)
    file_name_g = file_name_list[0] + "_g." + file_name_list[1]
    file_name_b = file_name_list[0] + "_b." + file_name_list[1]

    print("Wrote image"+ file_name_g)
    print("Wrote image" + file_name_b)
    cv2.imwrite(file_name_g, img_g)
    cv2.imwrite(file_name_b, img_b)

    sys.exit()
