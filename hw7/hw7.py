import warnings
import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans


'''
using ransac to get a set of inliers & outliers for FOE estimation
'''


def ransac(start, end):
    kmax = 0
    best_inlier = (0, 0)

    maxIter = 35
    vectors = end - start
    for i in range(maxIter):
        # generate two random indices
        sample = np.random.randint(0, vectors.shape[0], 2)
        ind1 = sample[0]
        ind2 = sample[1]
        if sample[0] == sample[1]:
            continue
        # find intersection of two vectors --> maybe FOE
        maybeFOE = lines_intersecting_point(start[ind1], end[ind1], start[ind2], end[ind2])

        # checks for invalid values possibly caused by inverting a singular matrix
        mask1 = np.isinf(maybeFOE)
        if np.any(mask1):
            print('inf caught')
            continue
        mask2 = np.isnan(maybeFOE)
        if np.any(mask2):
            print('nan caught')
            continue

        # checks to ensure FOE is in image plane --> that is the assumption
        if maybeFOE[0] < 0 or maybeFOE[1] < 0 or maybeFOE[0] > img2_col or maybeFOE[1] > img2_row:
            if maybeFOE[0] > img2_col or maybeFOE[1] > img2_row:
                # print('invalid FOE -- FOE outside image plane')
                maxIter -= 1
            continue

        # use this FOE to fit the rest of the lines
        # the line = vx-uy = x1v-y1u; with a = v, b = -u
        inliers = 0
        index = []
        for j in range(vectors.shape[0]):
            dist = lambda x, y: abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
            vfunc = np.vectorize(dist)
            a = vectors[j][1]  # v
            b = -vectors[j][0]  # -u
            c = end[j][0] * start[j][1] - start[j][0] * end[j][1]
            distance = vfunc(maybeFOE[0], maybeFOE[1])
            if distance < 3:  # allow for 10 pixel miss
                inliers += 1
                index.append(j)

        if inliers > kmax:
            kmax = inliers
            best_inlier = (maybeFOE[0], maybeFOE[1])
            inliers_start = []
            inliers_end = []
            outliers_start = []
            outliers_end = []
            out_index = [x for x in range(vectors.shape[0]) if x not in index]
            for u in index:
                inliers_start.append(start[u])
                inliers_end.append(end[u])
            for v in out_index:
                outliers_start.append(start[v])
                outliers_end.append(end[v])
            inliers_start = np.vstack(inliers_start)
            inliers_end = np.vstack(inliers_end)
            outliers_start = np.vstack(outliers_start)
            outliers_end = np.vstack(outliers_end)
    print('\t   Max number of inliers: %d\n' % kmax)
    print('\tFOE Estimate Using Ransac Best Inliers:', best_inlier)

    return inliers_start, inliers_end, outliers_start, outliers_end, best_inlier


'''
calculate the point of intersection
between two parametric line equations
'''


def lines_intersecting_point(p0, p1, q0, q1):
    mat = np.zeros((2, 2))
    mat[:, 0] = (p1 - p0)
    mat[:, 1] = -(q1 - q0)
    b = (q0 - p0)
    soln = np.dot(np.linalg.inv(mat), b)

    return p0 + soln[0] * (p1 - p0)


'''
draw motion vectors
'''


def arrows(Img, pt1, pt2, Color, thickness):
    assert (pt1.shape == pt2.shape)

    for i in range(pt1.shape[0]):
        Img = cv2.arrowedLine(Img, tuple(pt1[i]), tuple(pt2[i]), Color, thickness=thickness)

    return Img


'''
draw classified clusters and the bounding boxes
'''


def clusters(cluster_num, out_start, out_end, canvas, color):
    index = np.where(label_map == cluster_num)[0]
    cluster = []
    start = []
    for i in range(index.shape[0]):
        cluster.append(out_end[index[i]])
        start.append(out_start[index[i]])
    cluster = np.vstack(cluster)
    start = np.vstack(start)
    canvas = arrows(canvas, start, cluster, color, 3)
    min_x = int(np.amin(cluster, axis=0)[0])
    min_y = int(np.amin(cluster, axis=0)[1])
    max_x = int(np.amax(cluster, axis=0)[0])
    max_y = int(np.amax(cluster, axis=0)[1])
    canvas = cv2.rectangle(canvas, (min_x - 3, min_y - 3), (max_x + 3, max_y + 3), color, 2)

    return canvas


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    '''
    sys.stdout = open('out.txt', 'w')

    for i in range(24, 26):
    
        idx = str(i)
        print("Image index %s in progress..." %i)
        img1 = cv2.imread('data/' + idx + '_a.png')
        img2 = cv2.imread('data/' + idx + '_b.png')
        
    '''

    idx = str(sys.argv[1])
    img1 = cv2.imread('data/' + idx + '_a.png')
    img2 = cv2.imread('data/' + idx + '_b.png')

    copy = np.copy(img2)
    img2_row = img2.shape[0]
    img2_col = img2.shape[1]

    src1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    src2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # deep-learning-study.tistory.com/277
    # pt1 = cv2.goodFeaturesToTrack(src1, 25, 0.01, 10)
    # Shi-Tomasi
    pt1 = cv2.goodFeaturesToTrack(src1, mask=None,
                                  **dict(maxCorners=200, qualityLevel=0.6, minDistance=6, blockSize=3))


    # LK param
    pt2, status, err = cv2.calcOpticalFlowPyrLK(src1, src2, pt1, None, **dict(winSize=(8, 8),
                                                                              maxLevel=5,
                                                                              criteria=(
                                                                                  cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                                  10, 0.03)))

    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)


    print('\tNumber of keypoints obtained: %d' % pt1.shape[0])
    # remove status == 0, errors
    img1_flow = pt1[status == 1]
    img2_flow = pt2[status == 1]
    print('\tNumber of keypoints in motion %d' % img1_flow.shape[0])


    # thresholding
    vectors = img2_flow - img1_flow
    norms = np.linalg.norm(vectors, axis=1)

    lower_bound = 2  # number of pixels
    upper_bound = 10
    img1_flow = img1_flow[np.where(norms > lower_bound)]
    img2_flow = img2_flow[np.where(norms > lower_bound)]

    print('\tNumber of keypoints after thresholding %d\n' % img1_flow.shape[0])


    print('\tFinding inliers using RANSAC...')
    in_start, in_end, out_start, out_end, FOE = ransac(img1_flow, img2_flow)

    # number of inliers has to exceed certain number for the camera to be in motion
    num_of_inliers = in_start.shape[0]
    if num_of_inliers <= 4:
        # no motion detected
        print('\tNo camera motion detected because of not enough inliers (%d found)' %(in_start.shape[0]))
        outimg1 = img2
    else:
        # motion detected
        outimg1 = cv2.circle(img2, (int(FOE[0]), int(FOE[1])), 10, (0, 0, 255), thickness=-1)



    # green arrow: inliers, red arrow: outliers
    outimg1 = arrows(outimg1, in_start, in_end, (0, 255, 0), 3)
    outimg1 = arrows(outimg1, out_start, out_end, (0, 0, 255), 3)


    cv2.imwrite(idx + '_first.png', outimg1)
    '''
    cv2.imwrite('out/' + idx + '_first.png', outimg1)
    '''


    dst = arrows(dst, out_start, out_end, (0, 0, 255), 3)
    dst = arrows(dst, in_start, in_end, (0, 255, 0), 3)
    # cv2.imwrite('dst.png', dst)


    # k-mean clustering of outlier points
    kmeans = KMeans(n_clusters=3, random_state=0).fit(out_start)
    label_map = kmeans.labels_

    # one color for one cluster
    outimg2 = copy
    for j in range(3):
        color = np.random.uniform(low=0, high=255, size=3)
        outimg2 = clusters(j, out_start, out_end, outimg2, color)


    cv2.imwrite(idx + '_second.png', outimg2)
    '''
    cv2.imwrite('out/' + idx + '_second.png', outimg2)
    print("\n\n")


    sys.stdout.close()
    '''
