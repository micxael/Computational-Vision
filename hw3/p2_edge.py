# python p2_edge.py sigma_value input_image_name

import numpy as np
import cv2
import sys
import math


def encode_rgb(colors, degrees, magnitude):
    colors[(-pi / 8 <= degrees) & (degrees < pi / 8) & (magnitude > 1) & ((dx+dy) != 0), 2] = 255
    colors[(7*pi / 8 <= abs(degrees) ) & (abs(degrees) <= pi) & (magnitude > 1) & ((dx+dy) != 0), 2] = 255

    colors[(-7*pi / 8 <= degrees) & (degrees < -5*pi / 8) & (magnitude > 1) & ((dx+dy) != 0), 1] = 255
    colors[(pi / 8 <= degrees) & (degrees < 3*pi / 8) & (magnitude > 1) & ((dx+dy) != 0), 1] = 255

    colors[(-5*pi / 8 <= degrees) & (degrees < -3*pi / 8) & (magnitude > 1) & ((dx+dy) != 0), 0] = 255
    colors[(3*pi / 8 <= degrees) & (degrees < 5*pi / 8) & (magnitude > 1) & ((dx+dy) != 0), 0] = 255

    colors[(5*pi / 8 <= degrees) & (degrees < 7*pi / 8) & (magnitude > 1) & ((dx+dy) != 0), :] = 255
    colors[(-3*pi / 8 <= degrees) & (degrees <= -pi / 8) & (magnitude > 1) & ((dx+dy) != 0), :] = 255

    return colors


def encode_gradient(four_quad, degrees):
    four_quad[(-5*pi / 8 <= degrees) & (degrees < -3*pi / 8) & ((dx+dy) != 0)] = 1
    four_quad[(3*pi / 8 <= degrees) & (degrees < 5*pi / 8) & ((dx+dy) != 0)] = 1

    four_quad[(-7*pi / 8 <= degrees) & (degrees < -5*pi / 8) & ((dx+dy) != 0)] = 2
    four_quad[(pi / 8 <= degrees) & (degrees < 3*pi / 8) & ((dx+dy) != 0)] = 2

    four_quad[(-pi / 8 <= degrees) & (degrees < pi / 8) & ((dx+dy) != 0)] = 3
    four_quad[(7*pi / 8 <= abs(degrees)) & (abs(degrees) <= pi) & ((dx+dy) != 0)] = 3

    four_quad[(5*pi / 8 <= degrees) & (degrees < 7*pi / 8) & ((dx+dy) != 0)] = 4
    four_quad[(-3*pi / 8 <= degrees) & (degrees < -pi / 8) & ((dx+dy) != 0)] = 4

    return four_quad

def thresholding(mag_img, threshold):
    mag_img[np.where(threshold <= mag_img)] = 255
    mag_img[np.where(threshold > mag_img)] = 0

    return mag_img


def non_max_supp(image, four_quad, magnitude):
    temp = np.zeros((four_quad.shape[0], four_quad.shape[1]))
    mag = magnitude[1:-1, 1:-1]
    quad = four_quad[1:-1, 1:-1]
    r, c = magnitude.shape

    temp[2:r, 1:-1][(magnitude[0:-2, 1:-1] <= mag) & (magnitude[2:r, 1:-1] == mag) & (quad == 1)] = 1
    temp[0:-2, 1:-1][(magnitude[0:-2, 1:-1] == mag) & (magnitude[2:r, 1:-1] <= mag) & (quad == 1)] = 1
    temp[1:-1, 1:-1][(magnitude[0:-2, 1:-1] <= mag) & (magnitude[2:r, 1:-1] <= mag) & (quad == 1)] = 1

    temp[2:r, 2:c][(magnitude[0:-2, 0:-2] <= mag) & (magnitude[2:r, 2:c] == mag) & (quad == 2)] = 1
    temp[0:-2, 0:-2][(magnitude[0:-2, 0:-2] == mag) & (magnitude[2:r, 2:c] <= mag) & (quad == 2)] = 1
    temp[1:-1, 1:-1][(magnitude[0:-2, 0:-2] <= mag) & (magnitude[2:r, 2:c] <= mag) & (quad == 2)] = 1

    temp[1:-1, 1:-1][(magnitude[1:-1, 0:-2] <= mag) & (magnitude[1:-1, 2: c] <= mag) & (quad == 3)] = 1
    temp[1:-1, 2: c][(magnitude[1:-1, 0:-2] <= mag) & (magnitude[1:-1, 2: c] == mag) & (quad == 3)] = 1
    temp[1:-1, 0:-2][(magnitude[1:-1, 0:-2] == mag) & (magnitude[1:-1, 2: c] <= mag) & (quad == 3)] = 1

    temp[1:-1, 1:-1][(magnitude[2:r, 0:-2] <= mag) & (magnitude[0:-2, 2:c] <= mag) & (quad == 4)] = 1
    temp[0:-2, 2:c][(magnitude[2:r, 0:-2] <= mag) & (magnitude[0:-2, 2:c] == mag) & (quad == 4)] = 1
    temp[2:r, 0:-2][(magnitude[2:r, 0:-2] == mag) & (magnitude[0:-2, 2:c] <= mag) & (quad == 4)] = 1

    mag = np.zeros_like(image)
    mag[temp == 1] = magnitude[temp == 1]
    mag_post = np.zeros_like(mag)
    mag_post[mag > 1] = mag[mag > 1]
    non_max = len(np.where(mag != 0)[0])
    after_thresh = len(np.where(mag_post != 0)[0])

    average = np.average(mag_post[mag_post != 0])
    stn_dv = np.std(mag_post[mag_post != 0])
    threshold = min(average+0.5*stn_dv, 30/sigma)

    result = thresholding(mag_post, threshold)
    num_aft_thresh = len(np.where(result != 0)[0])

    print("Number after non-maximum: %d" % non_max)
    print("Number after 1.0 threshold: %d" % after_thresh)
    print("mu: %.2f" % average)
    print("s: %.2f" % stn_dv)
    print("Threshold: %.2f" % threshold)
    print("Number after threshold: %d" % num_aft_thresh)

    return result


if __name__ == '__main__':
    pi = math.pi
    sigma = float(sys.argv[1])
    fil_name = sys.argv[2]
    img_name = fil_name[:-4]
    fil_extension = fil_name[-4:]

    img = cv2.imread(fil_name, 0).astype(np.float)

    kernel = (int(sigma*4 + 1), int(sigma*4 + 1))
    blur = cv2.GaussianBlur(img, kernel, sigma)

    k_x, k_y = cv2.getDerivKernels(1, 1, 3)
    k_x = np.transpose(k_x / 2)
    k_y = k_y / 2

    dx = cv2.filter2D(blur, -1, k_x)
    dy = cv2.filter2D(blur, -1, k_y)

    d = dx ** 2 + dy ** 2
    magnitude = np.sqrt(d)

    norm_mag = magnitude / np.max(magnitude) * 255
    degrees = np.arctan2(dy, dx)
    cv2.imwrite('%s_grd%s' % (img_name, fil_extension), norm_mag)

    row, col = img.shape
    four_quad = np.zeros((row, col))
    color_img = np.zeros((row, col, 3))
    four_quad = encode_gradient(four_quad, degrees)
    color_rgb_img = encode_rgb(color_img, degrees, magnitude)

    cv2.imwrite('{}_dir{}'.format(img_name, fil_extension), color_rgb_img)

    result = non_max_supp(img, four_quad, magnitude)
    cv2.imwrite('{}_thr{}'.format(img_name, fil_extension), result)

    sys.exit()
