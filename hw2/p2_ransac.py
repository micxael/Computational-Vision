"""
Author:   Jae Park
Course:   CSCI 4270
HW2 Problem 2
File:     p2_ransac.py
python p2_ransac.py points.txt samples tau [seed]
Purpose:

"""

import cv2
import numpy as np
import os
import sys
import math
from matplotlib import pyplot as plt


def read_pts(f):
    pts = []
    for line in f:
        line = line.strip().split()
        x, y = [float(v) for v in line]
        pts.append([x, y])
    return np.array(pts)

def ransac(points, m, k, tau):

    a = -1 * m
    b = 1
    c = -1 * k

    k_max_new = 0
    in_dist_sum = 0
    out_dist_sum = 0
    for point in points:
        dist = abs(a * point[0] + b * point[1] + c) / math.sqrt(a**2 + b**2)
        if dist <= tau:
            k_max_new += 1
            in_dist_sum += dist
        else:
            out_dist_sum += dist

    return k_max_new, in_dist_sum, out_dist_sum

def print_line_info(m, k):
    n = math.sqrt(m**2 + 1)
    a = - m / n
    b = 1 / n
    c = - k / n
    print("line ({},{},{})".format(round(a, 3), round(b, 3), round(c, 3)))


# MAIN
if __name__ == '__main__':

    """
    Handle the command-line arguments
    """
    points_txt = sys.argv[1]
    sample_num = int(sys.argv[2])
    tau = float(sys.argv[3])

    if len(sys.argv) == 5:
        seed = int(sys.argv[4])
        np.random.seed(seed)

    file = open(points_txt)
    if file is None:
        print("Could not open %s" % points_txt)
        sys.exit()
    points = read_pts(file)

    sample = []
    N = len(points)
    for i in range(0, sample_num):
        sample.append(np.random.randint(0, N, 2))

    k_max = 0
    iter = 0
    latest_in_dist_sum = 0
    latest_out_dist_sum = 0
    for index in sample:
        if index[0] == index[1]:
            k_max_new = 0
        else:
            # RANSAC
            # form a line
            i = points[index[0]]
            j = points[index[1]]
            m = (j[1] - i[1]) / (j[0] - i[0])
            k = i[1] - m * i[0]


            # count how many kmax
            (k_max_new, in_dist_sum, out_dist_sum) = ransac(points, m, k, tau)


        # if kmax >, print infos
        if k_max_new > k_max:
            k_max = k_max_new

            print('Sample {}:'.format(iter))
            print('indices ({},{})'.format(index[0], index[1]))
            print_line_info(m, k)
            print('inliers', k_max, '\n')
            latest_in_dist_sum = in_dist_sum
            latest_out_dist_sum = out_dist_sum

        iter += 1
        # at the last iteration of the for loop
        if iter == len(sample):
            print()
            in_dist_avg = latest_in_dist_sum / k_max
            out_dist_avg = latest_out_dist_sum / (len(points) - k_max)
            print("avg inlier dist", format(round(in_dist_avg, 3), ".3f"))
            print("avg outlier dist", format(round(out_dist_avg, 3), ".3f"))

    sys.exit()
