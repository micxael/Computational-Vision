"""
Author:   Jae Park
Course:   CSCI 4270
Lecture: 03
File:     prob2_sol.py

Purpose: The script should read the
text file that represents a matrix. Then it finds and prints
the first principle component vector for the matrix.
"""

import sys
import numpy as np

def read_pts(f):
    pts = []
    for line in f:
        line = line.strip().split()
        x, y = [float(v) for v in line]
        pts.append([x, y])
    return np.array(pts)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s pts.txt" % sys.argv[0])
        sys.exit(0)
    f = open(sys.argv[1])
    if f is None:
        print("Could not open %s" % sys.argv[1])
        sys.exit()

    pts = read_pts(f)

    ###################
    (u, s, v) = np.linalg.svd(pts)

    if round(v[0, 0], 3) < 0:
        v[0, :] = v[0, :] * -1

    for i in v[0, :]:
        print(round(i, 3))

    sys.exit()
