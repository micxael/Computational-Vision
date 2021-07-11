"""
CSCI 4270/6270

This is the start to the exercise from Lecture 11 and 12 combined. The
main piece of code you need to write is the function map_corners
"""
import numpy as np
import sys
from numpy.linalg import multi_dot

def get_matrix_3x3(fp):
    values = []
    count = 0
    for line in fp:
        line = [float(a) for a in line.strip().split()]
        if len(line) != 3:
            print("Need three floats per input line.")
            sys.exit(1)
        values.append(line)
        count += 1
        if count == 3:
            return np.array(values)


def map_corners(K1, R1, K2, R2):

    upper_left = np.array([0, 0, 1])
    upper_right = np.array([0, 4000, 1])
    lower_left = np.array([6000, 0, 1])
    lower_right = np.array([6000, 4000, 1])

    H = multi_dot([K2, R2, np.transpose(R1), np.linalg.inv(K1)])

    u_2_upper_left = H.dot(upper_left)
    u_2_upper_right = H.dot(upper_right)
    u_2_lower_left = H.dot(lower_left)
    u_2_lower_right = H.dot(lower_right)

    u_2_upper_left = u_2_upper_left / u_2_upper_left[2]
    u_2_upper_right = u_2_upper_right / u_2_upper_right[2]
    u_2_lower_left = u_2_lower_left / u_2_lower_left[2]
    u_2_lower_right = u_2_lower_right / u_2_lower_right[2]


    print("%d %d" % (np.round(min(u_2_upper_left[0], u_2_upper_right[0], u_2_lower_left[0], u_2_lower_right[0])),
                     np.round(min(u_2_upper_left[1], u_2_upper_right[1], u_2_lower_left[1], u_2_lower_right[1])) ))
    print("%d %d" % (np.round(max(u_2_upper_left[0], u_2_upper_right[0], u_2_lower_left[0], u_2_lower_right[0])),
                     np.round(max(u_2_upper_left[1], u_2_upper_right[1], u_2_lower_left[1], u_2_lower_right[1])) ))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "matrices.txt")
        sys.exit(1)

    fp = open(sys.argv[1])
    if fp is None:
        print("Failed to open", sys.argv[1])
        sys.exit(1)

    K1 = get_matrix_3x3(fp)
    R1 = get_matrix_3x3(fp)
    K2 = get_matrix_3x3(fp)
    R2 = get_matrix_3x3(fp)

    map_corners(K1, R1, K2, R2)
    map_corners(K2, R2, K1, R1)
