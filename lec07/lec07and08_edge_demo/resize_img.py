import cv2
import numpy as np

def resize_img( im, max_dim ):
    scale = float(max_dim) / max(im.shape)
    if scale >= 1:
        return np.copy(im)

    new_size = (int(im.shape[1]*scale), int(im.shape[0]*scale))
    print(new_size)
    im_new = cv2.resize(im, new_size)   # creates a new image object
    return im_new


if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) != 4:
        print("Usage:", sys.argv[0], " in_img out_img max_dimension")
        sys.exit()
    in_name = sys.argv[1]
    out_name = sys.argv[2]
    max_dim = int(sys.argv[3])
    print("Scaling", in_name, "to max dimension of", max_dim)

    im = cv2.imread(sys.argv[1])
    im_new = resize_img(im, max_dim)
    print("Writing resized image to", out_name)
    cv2.imwrite(out_name, im_new)
