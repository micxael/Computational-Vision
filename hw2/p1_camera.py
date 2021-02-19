"""
Author:   Jae Park
Course:   CSCI 4270
HW2 Problem 2
File:     p1_camera.py
python p1_camera.py params.txt points.txt
Purpose:

"""

import cv2
import numpy as np
import sys


# rotational matrices
def rotation_matrix(x, rotation):
    if rotation == 'x':
        row_1 = np.array([1, 0, 0])
        row_2 = np.array([0, np.cos(x), -np.sin(x)])
        row_3 = np.array([0, np.sin(x), np.cos(x)])

    elif rotation == 'y':
        row_1 = np.array([np.cos(x), 0, np.sin(x)])
        row_2 = np.array([0, 1, 0])
        row_3 = np.array([-np.sin(x), 0, np.cos(x)])

    elif rotation == 'z':
        row_1 = np.array([np.cos(x), -np.sin(x), 0])
        row_2 = np.array([np.sin(x), np.cos(x), 0])
        row_3 = np.array([0, 0, 1])

    return np.vstack([row_1, row_2, row_3])


# create rotation matrix of r_x * r_y * r_z
def rotation(r_param):
    rotation_angle_x = r_param[0]
    rotation_angle_y = r_param[1]
    rotation_angle_z = r_param[2]
    x = rotation_matrix(rotation_angle_x, 'x')
    y = rotation_matrix(rotation_angle_y, 'y')
    z = rotation_matrix(rotation_angle_z, 'z')

    tmp = np.dot(y, z)
    return np.dot(x, tmp)


# combine rotational and translation matrix to form 3 x 4 matrix
def combine_rota_trans(r, t):
    temp = np.zeros((r.shape[0], r.shape[1] + 1))
    temp[:, 0:-1] = r
    temp[:, -1] = t
    return temp


# camera matrix M = K ( R_t - R_t * t)
def M(r, t, p):
    r_matrix = rotation(r)

    r_t_matrix = np.transpose(r_matrix)
    t_prime = -1 * np.dot(r_t_matrix, t)

    K_matrix = k_matrix(p)
    rt = combine_rota_trans(r_t_matrix, t_prime)

    M_matrix = np.dot(K_matrix, rt)

    return M_matrix


def k_matrix(p):
    f = p[0]
    # mm to microns
    d = p[1] * 0.001
    vc = p[2]
    uc = p[3]
    s = f / d
    row_1 = np.array([s, 0, uc])
    row_2 = np.array([0, s, vc])
    row_3 = np.array([0, 0, 1])
    return np.vstack([row_1, row_2, row_3])


# locate if a point is inside or outside the image plane
def locate(loc):
    col = loc[0]
    row = loc[1]
    if 0 <= row <= 4000 and 0 <= col <= 6000:
        print('%.1f %.1f inside' % (row, col))

    else:
        print('%.1f %.1f outside' % (row, col))


# determine if a point is in front of or behind the image plane
# using the dot product of plane normal and position vector
def visible(p_0, r, t):
    r_matrix = rotation(r)

    # camera center
    p = np.dot(r_matrix, np.array([0, 0, 0])) + t
    axis = np.dot(r_matrix, np.array([0, 0, 1]))

    # normalize the optical axis
    norm_axis = axis / np.linalg.norm(axis)
    val = np.inner(norm_axis, p_0 - p)

    # val > 0 : front & visible
    # else    : behind & hidden
    if val > 0:
        return True

    else:
        return False


# projects a point onto a plane
def projection(points, r, t):
    augment = combine_rota_trans(points, np.array([1] * points.shape[0]))
    t_point = np.transpose(augment)
    pxl_location = np.dot(camera_matrix, t_point)

    pxl_location[0] = pxl_location[0] / pxl_location[-1]
    pxl_location[1] = pxl_location[1] / pxl_location[-1]
    img_location = pxl_location[0:2, :]

    visible_idx = []
    hidden_idx = []

    print('Projections:')
    for x in range(img_location.shape[1]):
        str = '{}: {:.1f} {:.1f} {:.1f}'\
            .format(x, points[x][0], points[x][1], points[x][2])
        print(str, '=> ', end='')
        locate(img_location[:, x])

        ret = visible(points[x], r, t)
        if ret:
            visible_idx.append(x)
        else:
            hidden_idx.append(x)

    print('visible:', *visible_idx)
    print('hidden:', *hidden_idx)


def print_info(arr):
    for i in range(arr.shape[0]):
        print("{:.1f}, {:.1f}, {:.1f}, {:.1f}"\
              .format(arr[i][0], arr[i][1], arr[i][2], arr[i][3]))


# MAIN
if __name__ == "__main__":
    inputs = []
    file = open(sys.argv[1])

    for line in file:
        line = line.strip().split()
        inputs.append(line)

    r_param = np.array(inputs[0], dtype=np.float)
    r_param = np.deg2rad(r_param)

    t_param = np.array(inputs[1], dtype=np.float)
    params = np.array(inputs[2], dtype=np.float)

    camera_matrix = M(r_param, t_param, params)

    print('Matrix M:')
    print_info(camera_matrix)

    points = np.loadtxt(sys.argv[2])
    projection(points, r_param, t_param)

    sys.exit()
