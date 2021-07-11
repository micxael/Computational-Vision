import numpy as np
import cv2
import os
import sys


'''
image matching using SIFT features
'''

def matching(img1, img2, outImg):

    # parts of the code has been taken from OpenCV doc:
    # https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # queryImage
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # trainImage

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    matched_points1 = []
    matched_points2 = []

    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
            matched_points1.append(kp1[m.queryIdx].pt)
            matched_points2.append(kp2[m.trainIdx].pt)

    print("Keypoints from SIFT in {} : {}".format(file_1, len(kp1)))
    print("Keypoints from SIFT in {} : {}".format(file_2, len(kp2)))
    print("Keypoint matches: {}\n"
          .format(len(good)))

    good = np.array(good)
    matched_points1 = np.array(matched_points1)
    matched_points2 = np.array(matched_points2)

    out = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    file_output(outImg, out, '_BF', out_dir)

    fundamental_matrix(matched_points1, matched_points2, good, img1, img2, outImg)


'''
computing fundamental matrix estimate
'''

def fundamental_matrix(matched_points1, matched_points2, good, img1, img2, outImg):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    F, Mask = cv2.findFundamentalMat(matched_points1, matched_points2, cv2.FM_RANSAC, 1.0, 0.99)
    F_matches = good[Mask.ravel() == 1]

    print("Inliers consistent with F: {} (Ratio: {}% - inliers from F / BF matches)"
          .format((F_matches.shape[0]), round(F_matches.shape[0] * 100 / len(good)), 2))

    img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, F_matches, None, flags=2)
    file_output(outImg, img4, '_F', out_dir)

    # 20% threshold
    # sufficient for computing homography matrix
    if 0.2 * good.shape[0] < F_matches.shape[0]:
        print('Inliers ABOVE threshold: consistent with Fundamental Matrix\n')
        homography_matrix(img1, img2, matched_points1, matched_points2, good, outImg)
    else:
        print('Inliers BELOW threshold: inconsistent with Fundamental Matrix\n')


'''
computing homography matrix estimate
'''

def homography_matrix(image1, image2, matched_points1, matched_points2, good, out_img):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Homography Estimation
    H, Stat = cv2.findHomography(matched_points1, matched_points2, cv2.RANSAC,
                                 ransacReprojThreshold=1.0, confidence=0.99)
    h_matches = good[Stat.ravel() == 1]
    print("Inliers consistent with H: {} (Ratio: {}% - inliers from H / BF matches)" \
          .format((h_matches.shape[0]), round((h_matches.shape[0]) * 100 / len(good), 2)))

    img5 = cv2.drawMatchesKnn(image1, kp1, image2, kp2, h_matches, None, flags=2)
    file_output(out_img, img5, '_H', out_dir)

    # sufficient for computing mosaic
    if good.shape[0] * 0.3 < h_matches.shape[0]:
        print('Inliers ABOVE threshold: consistent with Homography Matrix\n')
        pick_img_4transformation(H, image1, image2, outName + '.jpg')
    else:
        print('Inliers BELOW threshold: inconsistent with Homography Matrix\n')


'''
compare between image 1 and image 2 using Homography matrix
to find image that is less distorted and 
use that to form the anchor of mosaic
'''

def pick_img_4transformation(H, img1, img2, outImgName):
    H_1 = np.sqrt(H[-1, 0] ** 2 + H[-1, 1] ** 2)
    H_2 = np.sqrt(np.linalg.inv(H)[-1, 0] ** 2 + np.linalg.inv(H)[-1, 1] ** 2)
    if H_1 < H_2:
        x_1, y_2 = offset(H, img1)
        composite = np.dot(np.array([[1, 0, x_1], [0, 1, y_2], [0, 0, 1]]), H)
        mosaic(composite, x_1, y_2, image1, image2, outImgName)
    else:
        x_2, y_2 = offset(np.linalg.inv(H), img1)
        composite = np.dot(np.array([[1, 0, x_2], [0, 1, y_2], [0, 0, 1]]), np.linalg.inv(H))
        mosaic(composite, x_2, y_2, image2, image1, outImgName)


'''
compute translational offset
'''

def offset(H, image1):
    x = 0
    y = 0
    t_l, l_r, l_l, t_r = mapping(H, np.array([0, 0, 1]), np.array([image1.shape[1], image1.shape[0], 1]),
                                 np.array([0, image1.shape[0], 1]), np.array([image1.shape[1], 0, 1]))
    if min([t_l[0], l_r[0], l_l[0], t_r[0]]) < 0:
        x = abs(min([t_l[0], l_r[0], l_l[0], t_r[0]]))
    if min([t_l[1], l_r[1], l_l[1], t_r[1]]) < 0:
        y = abs(min([t_l[1], l_r[1], l_l[1], t_r[1]]))
    return x, y


'''
form a mosaic image
given image 1 and image 2
and given matrix
'''

def mosaic(matrix, x, y, img1, img2, out_name):
    # new image1 coordinates
    t_l, l_r, l_l, t_r = mapping(matrix, np.array([0, 0, 1]), np.array([img1.shape[1], img1.shape[0], 1]),
                                 np.array([0, img1.shape[0], 1]), np.array([img1.shape[1], 0, 1]))
    # new image2 coordinates
    img2_t_l = (x, y)
    img2_l_l = (x, y + img2.shape[0])
    img2_t_r = (x + img2.shape[1], y)
    img2_l_r = (x + img2.shape[1], y + img2.shape[0])

    # mosaic size
    mosaic_row = max([t_l[1], l_r[1], l_l[1], t_r[1], img2_t_l[1], img2_l_r[1], img2_l_l[1], img2_t_r[1]])
    mosaic_col = max([t_l[0], l_r[0], l_l[0], t_r[0], img2_t_l[0], img2_l_r[0], img2_l_l[0], img2_t_r[0]])

    # mask generation and bi-linear interpolation
    result = cv2.warpPerspective(img1, matrix, (int(mosaic_col), int(mosaic_row)), flags=cv2.INTER_LINEAR)
    result2 = np.zeros(result.shape, dtype=result.dtype)
    result2[int(y):img2.shape[0] + int(y), int(x):img2.shape[1] + int(x)] = img2
    image1_ = np.zeros(result.shape, dtype=np.float32)
    image1_[np.where(result > 0)] = 1
    image2_ = np.zeros(result.shape, dtype=np.float32)
    image2_[int(y):img2.shape[0] + int(y), int(x):img2.shape[1] + int(x)] = 1

    # mask and weights combining
    matrix = image1_ + image2_
    mask1 = np.copy(image1_)
    mask1[np.where(matrix == 2)] = 0.20
    mask2 = np.copy(image2_)
    mask2[np.where(matrix == 2)] = 0.80
    image1_part = mask1 * result
    image2_part = mask2 * result2
    final_img = image1_part + image2_part
    file_output(out_name, final_img, '_mosaic', out_dir)

'''
new image coordinates for matrix A, B, C, D
consistent with matrix H transformation
'''

def mapping(H, A, B, C, D):
    A1 = np.dot(H, A)
    B1 = np.dot(H, B)
    C1 = np.dot(H, C)
    D1 = np.dot(H, D)
    return A1 / A1[-1], B1 / B1[-1], C1 / C1[-1], D1 / D1[-1]


'''
write image to 
/out_dir
'''

def file_output(filename, file, num, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_name = os.path.join(out_dir, filename + num + '.jpg')
    cv2.imwrite(file_name, file)


if __name__ == "__main__":
    file_list = []
    directory = sys.argv[1]
    out_dir = sys.argv[2]

    for file in os.listdir(directory):
        if file.lower().endswith('.jpg'):
            file_list.append(file)

    '''
    for each pair in the file list
    '''
    for i in range(0, len(file_list)):
        for j in range(i + 1, len(file_list)):
            file_1 = file_list[i].split('.')[0]
            file_2 = file_list[j].split('.')[0]
            image1 = cv2.imread(os.path.join(directory, file_list[i]))
            image2 = cv2.imread(os.path.join(directory, file_list[j]))
            print("[%s] and [%s] in process\n" % (file_1, file_2))
            outName = file_1 + '_' + file_2
            matching(image1, image2, outName)
            print("-----------------------------------------------------")

    sys.exit()
