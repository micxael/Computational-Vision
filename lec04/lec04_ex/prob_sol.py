"""
Lecture 4 Exercises
CSCI 4270
Jae Park

This is the start for the Lecture 4 problem.  Please modify 
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

    x_i = pts[:, 0]
    y_i = pts[:, 1]

    m_x = np.mean(x_i)
    m_y = np.mean(y_i)

    u_x = x_i - m_x
    u_y = y_i - m_y

    u_i_v_i = u_x * u_y

    u_x = np.square(u_x)
    u_y = np.square(u_y)

    M_matrix = [[np.sum(u_x), np.sum(u_i_v_i)], [np.sum(u_i_v_i), np.sum(u_y)]]

    M = np.array(M_matrix)

    w, v = np.linalg.eigh(M)

    # c = -1 * v[0, 0] * m_x - 1 * v[1, 0] * m_y

    if v[0, 0] < 0:
        v[:, 0] = -1 * v[:, 0]

    c = -1 * v[0, 0] * m_x - 1 * v[1, 0] * m_y

    print(np.round(v[0, 0], 3))
    print(np.round(v[1, 0], 3))
    print(np.round(c, 3))


    sys.exit()
