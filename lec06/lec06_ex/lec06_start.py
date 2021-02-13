"""
Lecture 06 Exercise
CSCI 4270 / 6270
Jae Park
"""

import sys
import numpy as np
import cv2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s image" % sys.argv[0])
        sys.exit(0)

    im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if im is None:
        print("Could not read image %s" % sys.argv[1])
        sys.exit(0)

    print(im)

    dx_kernel = np.array([[-0.5, 0, 0.5]])
    dy_kernel = np.array([[-0.5], [0], [0.5]])

    im_dx = cv2.filter2D(im, cv2.CV_32F, dx_kernel)
    im_dy = cv2.filter2D(im, cv2.CV_32F, dy_kernel)
    im_dx_abs = np.abs(2 * im_dx).astype(np.uint8)
    im_dy_abs = np.abs(2 * im_dy).astype(np.uint8)

    print()

    sys.exit()
