"""
Author:   Jae Park
Course:   CSCI 4270
HW1 Problem 3
File:     p3_shade.py
python p3_shade.py input.jpg output.jpg "dir"
Purpose:

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import sys


def calculate_weight(input_img, dir):

    M = input_img.shape[0]
    N = input_img.shape[1]

    if(dir == 'left'):
        arr = np.arange(0, N)
        arr = np.flip(arr)
        arr = arr / (N - 1)
        arr = np.tile(arr, (M, 1))

    elif(dir == 'right'):
        arr = np.arange(0, N)
        arr = arr / (N - 1)
        arr = np.tile(arr, (M, 1))

    elif (dir == 'top'):
        arr = np.arange(0, M)
        arr = arr / (M-1)
        arr = np.flip(arr)
        arr = arr.reshape(-1, 1)
        arr = np.tile(arr, (1, N))

    elif (dir == 'bottom'):
        arr = np.arange(0, M)
        arr = arr / (M - 1)
        arr = arr.reshape(-1, 1)
        arr = np.tile(arr, (1, N))

    else: #center
        arr = center_weighted_array( M, N)

    weight_output(arr, M, N)
    return arr


def center_weighted_array(M, N):

    M_arr = np.zeros(M)
    m = M // 2
    N_arr = np.zeros(N)
    n = N // 2

    for i in range(0, M):
        M_arr[i] = m - i
    for i in range(0, N):
        N_arr[i] = n - i

    M_arr = M_arr.reshape(-1, 1)

    M_arr = np.tile(M_arr, (1, N))
    N_arr = np.tile(N_arr, (M, 1))

    M_arr = np.square(M_arr)
    N_arr = np.square(N_arr)

    arr = M_arr + N_arr
    arr = np.sqrt(arr)

    # normalize_factor = arr[0][0]
    normalize_factor = np.sqrt(np.square(m) + np.square(n))

    arr = arr / normalize_factor

    arr = 1 - arr

    return arr


def weight_output(arr, M, N):
    # np.set_printoptions(precision=3)
    print('({},{}) {}' .format('0', '0', format(arr[0][0], ".3f")))
    print('({},{}) {}'.format('0', N//2, format(arr[0][N//2], ".3f")))
    print('({},{}) {}'.format('0', N-1, format(arr[0][N-1], ".3f")))
    print('({},{}) {}'.format(M//2, '0', format(arr[M//2][0], ".3f")))
    print('({},{}) {}'.format(M//2, N//2, format(arr[M//2][N//2], ".3f")))
    print('({},{}) {}'.format(M//2, N-1, format(arr[M//2][N-1], ".3f")))
    print('({},{}) {}'.format(M-1, '0', format(arr[M-1][0], ".3f")))
    print('({},{}) {}'.format(M-1, N//2, format(arr[M-1][N//2], ".3f")))
    print('({},{}) {}'.format(M-1, N-1, format(arr[M-1][N-1], ".3f")))


# MAIN
if __name__ == '__main__':

    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 4:
        print('Arguments not provided correctly.')
        sys.exit()

    in_img_name = sys.argv[1]
    out_img_name = sys.argv[2]
    direction = sys.argv[3]

    input_img = cv2.imread(in_img_name)

    weight_array = calculate_weight(input_img, direction)

    weight_array = np.dstack((weight_array, weight_array, weight_array))

    out_img = input_img * weight_array

    out_img = np.concatenate((input_img, out_img), axis=1)
    cv2.imwrite(out_img_name, out_img)

    sys.exit()
