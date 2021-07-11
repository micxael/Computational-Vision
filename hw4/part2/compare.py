import numpy as np
import cv2
import sys


'''
Matching with SIFT Descriptors and symmetric matching
'''
def symmetric_matching(img1_gray, img2_gray):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)


    # cv2.NORM_L2SQR: sqrt(sum(a-b)**2))
    # other variations such as cv2.NORM_L1, cv2.NORM_L2

    # crossCheck=True is better for ratio test
    # according to David G. Lowe on SIFT algorithm
    bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=False)


    # Brute Force match the descriptors from img1 & img2,
    # comparing them based on closest Euclidean distances
    matches = bf.match(des1, des2)


    # only display the best 90% of the matches
    # by sorting the matching descriptors in the order of their distance
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    print("keypoints from symmetric distance matching: %d" % (len(sorted_matches)))
    num_matches_threshold = int(0.9 * len(sorted_matches))
    output_img = cv2.drawMatches(img1, kp1, img2, kp2,
                                 sorted_matches[:num_matches_threshold], None, flags=2)


    cv2.imwrite("SIFT_symmetric_matching.jpg", output_img)
    cv2.imshow("SIFT_symmetric_matching", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # organize keypoints to find fundamental matrix
    good = []
    matched_points1 = []
    matched_points2 = []
    for m in sorted_matches:
        good.append([m])
        matched_points1.append(kp1[m.queryIdx].pt)
        matched_points2.append(kp2[m.trainIdx].pt)


    find_fundamental_matrix(matched_points1, matched_points2, good, "symmetric")


'''
Matching with SIFT Descriptors and Ratio Test matching
'''
def ratio_test(img1_gray, img2_gray):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    # k best matches
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    matched_points1 = []
    matched_points2 = []
    threshold = 0.85

    for m, n in matches:
        if m.distance / n.distance < threshold:
            good.append([m])
            matched_points1.append(kp1[m.queryIdx].pt)
            matched_points2.append(kp2[m.trainIdx].pt)

    print("keypoints from Ratio Test matching: %d" % (len(good)))
    output_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)


    cv2.imwrite("SIFT_ratio_test.jpg", output_img)
    cv2.imshow("SIFT_ratio_test", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    find_fundamental_matrix(matched_points1, matched_points2, good, "ratio")


'''
find number of inliers that are consistent with the fundamental matrix
given all the keypoints previously found
'''
def find_fundamental_matrix(matched_points1, matched_points2, good, option):
    good = np.array(good)
    matched_points1 = np.array(matched_points1)
    matched_points2 = np.array(matched_points2)

    F, Mask = cv2.findFundamentalMat(matched_points1, matched_points2, cv2.FM_RANSAC)
    F_matches = good[Mask.ravel() == 1]

    if option == "symmetric":
        inlier_threshold = 0.2
    elif option == "ratio":
        inlier_threshold = 0.5

    if inlier_threshold * good.shape[0] < F_matches.shape[0]:
        print("Inliers consistent with F: {} (Ratio: {}%)"
              .format((F_matches.shape[0]), round(F_matches.shape[0] * 100 / len(good)), 2))
        print('Inliers ABOVE threshold {}%: consistent with Fundamental Matrix\n'.format(inlier_threshold * 100))
    else:
        print("Inliers consistent with F: {} (Ratio: {}%)"
              .format((F_matches.shape[0]), round(F_matches.shape[0] * 100 / len(good)), 2))
        print('Inliers BELOW threshold {}%: inconsistent with Fundamental Matrix\n'.format(inlier_threshold * 100))


if __name__ == "__main__":
    img1 = sys.argv[1]
    img2 = sys.argv[2]

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # queryImage
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    symmetric_matching(img1_gray, img2_gray)
    ratio_test(img1_gray, img2_gray)

    sys.exit()
