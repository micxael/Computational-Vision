"""
Author:   Chuck Stewart
Course:   CSCI 4270 and 6270
Lecture:  02
File:     ex5_stretch_img.py

Purpose: Linearly stretch the gray scale intensities of an input image
based on the cutoff fraction, f, provided on the command-line.  First
we find the maximum intensity --- call it g0 --- such that f of the
pixels have intensity less than or equal to g0.  Then we find the
minimum intensity --- call it g1 --- such that 1-f of the pixels in
the image have intensity greater than or equal to g1.  Then we map the
intensities as follows:

  . All intensities <= g0 are mapped to 0
  . All intensitiess >= g1 are mapped to 255
  . Intensities from g0 to g1 are linearly stretched into the range
    [0,255]

NumPy:  This demonstrates the use of NumPy histograms, cumulative
distributions, the where function, indexing with a boolean array, and
image arithmetic.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


def show_with_pixel_values(im):
    plt.axis("off")
    plt.imshow(im)
    plt.show()
    return

    """
    Create a plot of a single image so that pixel intensity or color
    values are shown as the mouse is moved over the image.

    This is a tool everyone can use, but it is not the main focus of
    this example
    """
    fig, ax = plt.subplots()
    numrows, numcols = im.shape[0], im.shape[1]
    plt.gray()
    ax.imshow(im)

    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = im[row, col]
            if type(z) is np.ndarray:    # color image
                return '(%d,%d): [%d,%d,%d]' \
                    % (col, row, int(z[0]), int(z[1]), int(z[2]))
            else:                       # grayscale image
                return '(%d,%d): %d' % (x, y, int(z))
        else:
            return '(%d,%d)' % (x, y)

    ax.format_coord = format_coord  
    plt.show()


if __name__ == "__main__":
    '''
    Now onto the real stuff.  First, check the command line arguments.
    '''
    if len(sys.argv) not in [3, 4]:
        print("Usage: %s image-file cutoff-frac [-v]" % sys.argv[0])
        sys.exit()

    '''
    Open and read the image.
    '''
    im = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if im is None:
        print("Failed to open image", sys.argv[1])
        sys.exit()

    '''
    Get the cutoff fraction f and make sure it is in range (0,0.5)
    '''
    f = float(sys.argv[2])
    if f < 0.0 or f >= 0.5:
        print("The 2nd argument, the cutoff fraction, must be in [0, 0.5)")
        sys.exit()

    '''
    The first half of the real work: Form the histogram, convert it to
    cumulative distribution, and find the g0 and g1 cutoffs, all in 6
    lines of Python NumPy code!
    '''
    hist, bins = np.histogram(im, bins=256, range=(0, 255))
    cum_dist = np.cumsum(hist) / np.sum(hist)
    below_cutoff = np.where(cum_dist <= f)[0]
    g0 = below_cutoff[-1]
    above_cutoff = np.where(cum_dist >= 1-f)[0]
    g1 = above_cutoff[0]

    '''
    If verbose output is requested, this is shows the histogram, the
    cumulative distribution, and the selected g0 and g1.
    '''
    if "-v" in sys.argv:
        print("Histogram")
        for i, v in enumerate(hist):
            print("\t%3d: %d" % (i, v))
        print()
        print("Cumulative distribution")
        for i, v in enumerate(cum_dist):
            print("\t%3d: %f" % (i, v) )
        print()
        print("g0 %d, g1 %d" % (g0, g1))

    '''
    Here is the second half of the real work:  compute the scaled
    image by truncating values below g0 at g0, truncating values above
    g1 at g1, and linearly scaling the result into the range 0..255.
    The procedure is slightly different from what was described in the
    introductory comments, but the result is the same.
    '''
    im_scaled = im.copy()    # requires a deep copy
    im_scaled[im_scaled < g0] = g0
    im_scaled[im_scaled > g1] = g1
    im_scaled = (im_scaled - g0) / (g1 - g0) * 255
    im_scaled = im_scaled.astype(im.dtype)

    '''
    Show the two images side-by-side.  vmin and vmax are set so that
    imshow does not automatically scale the images.
    '''
    plt.gray()
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(im, vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(im_scaled, vmin=0, vmax=255)
    plt.show()

    '''
    Show the images individually with their pixel values.
    '''
    show_with_pixel_values(im)
    show_with_pixel_values(im_scaled)
