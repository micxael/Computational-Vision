"""
Lecture 06 Exercise
CSCI 4270 / 6270
Jae Park
"""

import sys
import numpy as np
import cv2

# python lec06_ex.py image_name.jpg
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s image" % sys.argv[0])
        sys.exit(0)

    im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if im is None:
        print("Could not read image %s" % sys.argv[1])
        sys.exit(0)

    # Discrete gradient method of weights (-1/2, 0, 1/2)
    dx_kernel = np.array([[-0.5, 0, 0.5]])
    dy_kernel = np.array([[-0.5], [0], [0.5]])

    im_dx = cv2.filter2D(im, cv2.CV_32F, dx_kernel)
    im_dy = cv2.filter2D(im, cv2.CV_32F, dy_kernel)

    gradient_arr = np.sqrt(np.square(im_dx) + np.square(im_dy))

    avg1 = np.average(gradient_arr)

    print('Two-valued average:', round(avg1, 1))

    # Sobel method
    im_dx = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    im_dx = im_dx / 8

    im_dy = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    im_dy = im_dy / 8

    Sobel_gradient = np.sqrt(np.square(im_dx) + np.square(im_dy))
    avg2 = np.average(Sobel_gradient)
    print('Sobel average:', round(avg2, 1))


    # avg of absolute difference between two gradient methods
    diff = gradient_arr - Sobel_gradient
    abs_diff = np.abs(diff)
    print('Average diff:', round(np.average(abs_diff), 1))


    # The maximum difference between the gradients for pixels
    # where the two-valued gradient magnitude is larger
    print('Max two-valued diff:', round(np.max(diff), 1))

    # The maximum difference between the gradients for pixels
    # where the Sobel gradient magnitude is larger
    print('Max Sobel diff:', round(-1 * np.min(diff), 1))

    count = 0
    for x in diff:
        for y in x:
            if y > 0:
                count += 1
    percentage = count / (diff.shape[0] * diff.shape[1]) * 100
    print('Pct two-valued greater:', round(percentage))

    sys.exit()
