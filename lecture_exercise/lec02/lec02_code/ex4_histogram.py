"""
Author:   Chuck Stewart
Course:   CSCI 4270 and 6270
Lecture:  02
File:     ex4_histogram.py

Purpose:  Show an image side-by-side with its histogram.  Demonstrates
use of gray-scale conversion, shape and type changes, calculation of
histgrams, and non-image display in MatPlotLib.

Some of this code is adapted from:
http://docs.opencv.org/trunk/d1/db7/tutorial_py_histogram_begins.html
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# Check the usage
if len(sys.argv) not in [2, 3]:
    print("Usage: %s image-file [-g]" % sys.argv[0])
    print("where -g requests a gray-scale histogram; color is the default")
    sys.exit()

#  Read the image.  If imread fails it will return None
im = cv2.imread(sys.argv[1])
if im is None:
    print("Failed to open image", sys.argv[1])

# Determine if the user requested a gray scale histogram
plot_gray = len(sys.argv) == 3 and sys.argv[2] == '-g'

"""
If a gray scale histogram is requested and the image is color, then
it must be converted to gray scale.
"""
if plot_gray and len(im.shape) == 3:
    # Convert a color image to gray scale for histogram calculation
    print('Converting to gray scale')
    print('Shape before conversion is', im.shape)

    """
    # Here is NumPy code for gray scale conversion/
    # Note that the image is still BGR, which means the
    # 0.299 weight on red is applied to entry 2 in each pixel
    im1 = 0.299*im[:,:,2] + 0.587*im[:,:,1] + 0.114*im[:,:,0]
    im = im1.astype(im.dtype)
    """

    """
    # Here is alternative NumPy code for gray scale conversion.
    # Because the image is still BGR, the vector is created
    # with the blue wgt first and the red weight last
    wgt_vect = np.array([0.114, 0.587, 0.299])
    im1 = np.dot(im, wgt_vect)
    im = im1.astype(im.dtype)
    """
    # This is the OpenCV code for gray scale conversion
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    print('Shape after conversion is', im.shape)

elif len(im.shape) == 3:
    # Otherwise, convert BGR to RGB
    im = im[:, :, ::-1]

# Get the number of channels to form the histogram
if len(im.shape) == 3:
    num_channels = im.shape[2]  # should be 3
else:
    num_channels = 1

# Plot the image on the left side.
plt.subplot(121)
plt.axis("off")
plt.gray()
plt.imshow(im)

'''
Now generate the plot according to whether a one-channel or
three-channel histogram is needed.
'''
plt.subplot(122)
num_pixels = im.shape[0] * im.shape[1]
if num_channels == 3:
    color = ('r', 'g', 'b')
    max_hist = 0
    for channel, col in enumerate(color):
        histr = cv2.calcHist([im], [channel], None, [256], [0, 256])
        # print(histr)
        histr *= 100 / num_pixels   # pct of the total number of pixels
        max_hist = max(np.max(histr), max_hist)
        plt.plot(histr, color=col)
else:
    histr = cv2.calcHist([im], [0], None, [256], [0, 256])
    histr *= 100 / num_pixels     # pct of the total number of pixels
    max_hist = np.max(histr)
    plt.plot(histr)

'''
Generate the histogram plot, scaling the max value by 10% to make a
clean upper edges of the histogram.
'''
plt.title('Intensity histogram in % pixels')
max_hist *= 1.1
plt.ylim([0, max_hist])
plt.xlim([0, 256])
plt.show()
