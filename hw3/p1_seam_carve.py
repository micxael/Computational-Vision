#python p1_seam_carve.py input_image_name

import numpy as np
import cv2
import sys


def sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    return np.absolute(sobel_x) + np.absolute(sobel_y)


'''apply the seam carve'''
def seam_carving(sobel_img, color_img, gray_img, count, direction):

    W = sobel_img.copy()
    W[:, 0] = 0
    W[:, color_img.shape[1] - 1] = 0
    W[:, 0] = 100000
    W[:, -1] = 100000

    for i in range(1, color_img.shape[0]):
        prev_weight = W[i - 1, :]
        prev_line_min = np.zeros((gray_img.shape[1]))

        left = prev_weight[:-2]
        right = prev_weight[2:]
        center = prev_weight[1:-1]

        a = (1-((center == right)*(center == left))) *\
            (center <= right) * (center <= left) * center
        b = ((1-((center == right)*(center == left))) *\
            (right <= center) * (right <= left)) * right
        c = ((1-((center == right)*(center == left))) *\
            (left <= right) * (left <= center)) * left
        d = ((center == right) * (center == left)) * left

        prev_line_min[1:-1] = a + b + c + d

        prev_line_min[0] = min(prev_weight[0], prev_weight[1])
        prev_line_min[-1] = min(prev_weight[-1], prev_weight[-2])
        W[i, :] += prev_line_min


    trace = []
    last_min = np.argmin(W[color_img.shape[0] - 1, :])
    tracing(W, last_min, color_img.shape[0] - 1, trace)

    is_max = np.zeros_like(gray_img, dtype=np.bool)
    np.ndarray.fill(is_max, True)

    trace.reverse()
    energy_sum = 0
    tmp = color_img.copy()

    for index in range(len(trace)):
        x = index
        y = trace[index]
        tmp[x, y] = [0, 0, 255]
        is_max[x, y] = False
        energy_sum += sobel_img[x, y]
        if count == 0 or count == 1 or count == diff - 1:
            if index == 0 or index == len(trace)//2 or index == len(trace)-1:
                if direction == 0:
                    print("%d , %d" %(x, y))
                else:
                    print("%d , %d" %(y, x))

    if count == 0:
        if direction == 1:
            tmp = np.rot90(tmp, 3)
        cv2.imwrite('{}_seam.jpg'.format(img_name), tmp)

    color_mask = np.repeat(is_max[:, :, np.newaxis], 3, axis=2)
    if count == 0 or count == 1 or count == diff-1:
        print('Energy of seam {}: {:.2f}\n'.format(count, energy_sum / len(trace)))

    new_color = color_img.copy()
    new_color = new_color[color_mask]
    new_color = new_color.reshape((color_img.shape[0], color_img.shape[1] - 1, 3))
    new_gray = gray_img.copy()
    new_gray = new_gray[is_max]
    new_gray = new_gray.reshape((color_img.shape[0], color_img.shape[1] - 1))

    return new_color, new_gray


def tracing(M, prev_min, curr_row, trace_arr):
    if curr_row < 0:
        return
    trace_arr.append(prev_min)
    #trace_arr += prev_min

    if prev_min == 0:
        index = np.argmin(M[curr_row - 1, 0: 2])
        index = prev_min + index
        tracing(M, index, curr_row - 1, trace_arr)
    elif prev_min == M.shape[1] - 1:
        index = np.argmin(M[curr_row - 1, prev_min - 1: M.shape[1]])
        index = prev_min + (1 - index)
        tracing(M, index, curr_row - 1, trace_arr)
    # general case
    else:
        index = np.argmin(M[curr_row - 1, prev_min - 1:prev_min + 2])
        index = prev_min + (index - 1)
        tracing(M, index, curr_row - 1, trace_arr)


def pretty_print(count, direction):
    print('Points on seam %d:' % count)
    if direction == 0:
        print('vertical')
    else:
        print('horizontal')


if __name__ == '__main__':
    file_name = sys.argv[1]
    img_name = file_name[:-4]
    img = cv2.imread(file_name).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col, row = gray.shape

    if row > col:
        diff = row - col
        gray_carving = gray
        color_carving = img
        direction = 0
    else:
        diff = col - row
        gray_carving = np.rot90(gray)
        color_carving = np.rot90(img)
        direction = 1

    count = 0
    for i in range(diff):
        if count == 0 or count == 1 or count == diff-1:
            pretty_print(count, direction)

        sobel_img = sobel(gray_carving)
        color_carving, gray_carving =\
            seam_carving(sobel_img, color_carving, gray_carving, count, direction)
        count += 1
    #end_for

    if row < col:
        color_carving = np.rot90(color_carving, 3)

    cv2.imwrite('{}_final.jpg'.format(img_name), color_carving)

    sys.exit()
