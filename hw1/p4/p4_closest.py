"""
Author:   Jae Park
Course:   CSCI 4270
HW1 Problem 4
File:     p4_closest.py
python p4_closest.py img_folder
Purpose:

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

import os
import sys


def jpg_filename_from_directory(dir):
    images_list = os.listdir(dir)

    for filename in images_list:
        if not filename.endswith(".jpg"):
            images_list.remove(filename)

    return images_list


def load_images_from_folder(dir, images_list):
    images = []
    for filename in images_list:
        img = cv2.imread(os.path.join(dir, filename))
        if img is not None:
            images.append(img)

    return images


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

    downsized_img = np.zeros((m, n, 3))

    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, 3):
                downsized_img[i, j, k] = \
                    np.average(img[i_index[i]:i_index[i + 1], j_index[j]:j_index[j + 1], k])

    return downsized_img


def normalize_img(img, option):
    if option == "rgb":
        img = np.ravel(img)
        img = img / np.linalg.norm(img, keepdims=True)

        img = img * 100

    else: #Lab
        img = np.ravel(img)
        L_val_avg = np.average(img[0::3])
        img[0::3] = img[0::3] - L_val_avg + 128

    return img


def calc_distances(arr, images_list):
    n = len(arr)

    for i in range(0, n):
        distances = []
        for j in range(0, n):
            if i == j:
                distances.append(float('inf'))
            else:
                distances.append(
                    distance_two_vecs(arr[i], arr[j]))
        min_index = distances.index(min(distances))
        print('{} to {}: {}'.format(images_list[i], images_list[min_index],
                                    round(distances[min_index], 2)))



def distance_two_vecs(arr1, arr2):
    sub_arr = arr1 - arr2
    sub_arr = np.square(sub_arr)

    distance = np.sqrt(sum(sub_arr))

    return distance


# MAIN
if __name__ == '__main__':

    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 3:
        print('Arguments not provided correctly.')
        sys.exit()

    dir = sys.argv[1]
    n = int(sys.argv[2])

    images_list = jpg_filename_from_directory(dir)

    images_list = sorted(images_list)
    images_rgb = load_images_from_folder(dir, images_list)

    images_Lab = []
    for img in images_rgb:
        images_Lab.append(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

    down_norm_rgb = []
    down_norm_Lab = []

    for img in images_rgb:
        img = downsized_img(img, n, n)
        img = normalize_img(img, "rgb")
        down_norm_rgb.append(img)

    for img in images_Lab:
        img = downsized_img(img, n, n)
        img = normalize_img(img, "Lab")
        down_norm_Lab.append(img)

    print('Returning images:', images_list)
    print('RGB nearest distances')
    print('First region: {} {} {}'
          .format(round(down_norm_rgb[0][2], 3),
                  round(down_norm_rgb[0][1], 3),
                  round(down_norm_rgb[0][0], 3)))

    print('Last region: {} {} {}'
          .format(round(down_norm_rgb[0][-1], 3),
                  round(down_norm_rgb[0][-2], 3),
                  round(down_norm_rgb[0][-3], 3)))

    calc_distances(down_norm_rgb, images_list)

    print('\nL*a*b nearest distances')
    print('First region: {} {} {}'
          .format(round(down_norm_Lab[0][0], 3),
                  round(down_norm_Lab[0][1], 3),
                  round(down_norm_Lab[0][2], 3)))

    print('Last region: {} {} {}'
          .format(round(down_norm_Lab[0][-3], 3),
                  round(down_norm_Lab[0][-2], 3),
                  round(down_norm_Lab[0][-1], 3)))

    calc_distances(down_norm_Lab, images_list)

    sys.exit()
