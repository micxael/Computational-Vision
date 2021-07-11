"""
Lecture 5 Exercises
CSCI 4270 / 6270
Prof. Stewart

This is the start for the Lecture 5 problem.  Please modify 
"""

import sys
import numpy as np


def read_matrices(f):
    """
    Get a list of 3x3 matrics from f.  Blank lines are allowed, but
    each line of numbers must have exactly three floats and nothing else.
    """
    matrices = []
    pts = []
    
    for line in f:
        line = line.strip().split()
        if len(line) == 0:
            continue
        line = [float(v) for v in line]
        assert(len(line) == 3)
        pts.append(line)
        if len(pts) == 3:
            h = np.array(pts)
            pts = []
            h /= h[2, 2]
            matrices.append(h)

    assert(len(pts) == 0)  # No extra values
    return matrices


def is_equal(a, b):
    return abs(a-b) < 1.e-6


def assess_matrix(matrix):
    # organize the matrix into upper-right triangle
    # matrix = np.triu(matrix, -1)

    # rigid or similarity transformation
    if matrix[0][0] == matrix[1][1]:
        if matrix[1][0] == -1 * matrix[0][1]:
            one = np.square(matrix[0][0]) + np.square(matrix[1][0])
            if round(one, 5) == 1:
                print("rigid")
                return
            elif round(one, 5):
                print("similarity")
                return

    sub_matrix = matrix[0:2, 0:2]

    if np.linalg.det(sub_matrix) != 0:
        print("affine")
        return

    count = matrix.shape[0] # row size
    for i in range(count):
        # if (0, 0, n) row exists
        # i.e., A is not a valid 2 x 2 matrix
        if matrix[i, 0] == 0 and matrix[i, 1] == 0:
            count -= 1

    # A is a valid 2 x 2 matrix
    if count == matrix.shape[0] - 1:
        print("homography")
        return
    else:
        print("none")
        return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s matrix.txt" % sys.argv[0])
        sys.exit(0)
    f = open(sys.argv[1])
    if f is None:
        print("Could not open %s" % sys.argv[1])
        sys.exit()

    matrices = read_matrices(f)

    for i in range(0, len(matrices)):
        assess_matrix(matrices[i])

    sys.exit()
