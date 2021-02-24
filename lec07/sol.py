
import sys

import numpy as np
import cv2
from PIL import Image



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: %s in_image out_image sigma g0 g1" % sys.argv[0])
        sys.exit(0)

    im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if im is None:
        print("Could not read image %s" % sys.argv[1])
        sys.exit(0)
    out_name = sys.argv[2]
    sigma = int(sys.argv[3])
    g0 = float(sys.argv[4])
    g1 = float(sys.argv[5])


    # image smoothing
    ksize = (4 * sigma + 1, 4 * sigma + 1)
    im_s = cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)

    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2
    # print('Here are the derivative kernels')
    # print(kx)
    # print(ky)


    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)
    im_gm = np.sqrt(im_dx ** 2 + im_dy ** 2)  # gradient magnitude


    # normalize by gradient magnitude
    im_gm = im_gm + 0.00001
    im_dx = im_dx / im_gm
    im_dy = im_dy / im_gm
    '''
    im_dx = np.nan_to_num(im_dx)
    im_dy = np.nan_to_num(im_dy)
    im_dx[im_dx == 0] = 0.00001
    im_dy[im_dy == 0] = 0.00001
    '''


    # map -1 to 0, 1 to 255
    im_dx = np.round((im_dx + 1) * (255 / 2))
    im_dy = np.round((im_dy + 1) * (255 / 2))


    # thresholding
    im_gm[im_gm < g0] = 0
    im_gm[im_gm > g1] = g1
    im_gm = np.round(im_gm * (255 / g1))


    # whetr gradient is g0 or less
    for (x, y), val in np.ndenumerate(im_gm):
        if val <= g0:
            im_dx[x, y] = 0
            im_dy[x, y] = 0


    image = np.dstack((im_gm, im_dx, im_dy))
    cv2.imwrite(out_name, cv2.cvtColor(image, cv2.COLOR_Lab2BGR))

    '''
    I'm not really sure where I got this wrong,
    but somehow I'm getting only a dark image...
    '''

    sys.exit()
