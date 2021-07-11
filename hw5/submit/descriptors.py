import numpy as np
import cv2
import os
import pickle


def generate_descriptor(img, bin_num, b_h, b_w, block_H, block_W, u_l, l_r, label):
    # generate descriptor for an image img

    # caclulate all the image block indices
    # u_l: upper left corner, l_r: lower right corner
    blocks = [img[u_l[i, 0]:l_r[i, 0], u_l[i, 1]:l_r[i, 1], :] for i in range(b_h * b_w)]

    # Calculate the 3d histogram
    pixels = [im.reshape(block_H * block_W, 3) for im in blocks]
    histograms = [np.histogramdd(pixel, (bin_num, bin_num, bin_num))[0] for pixel in pixels]

    # form a one contiguous descriptor vector
    desc = np.ravel([np.ravel(histogram) for histogram in histograms])
    # append image label
    desc = np.concatenate((desc, np.array([label])))

    return desc


def desc_vectors(images, out_name, label):
    # generates aggregated descriptor for an image folder
    descriptors_list = []

    for img in images:
        row, col = img.shape[0], img.shape[1]

        # parameters
        bin_num = 4
        # blocks
        b_h = 4
        b_w = b_h
        H = row / (b_h + 1)
        W = col / (b_w + 1)
        block_H = int(2 * H)
        block_W = int(2 * W)

        # upper-left and lower-right borders of the blocks
        u_l = np.array([(n * H, m * W) for n in range(b_h) for m in range(b_w)]).astype(np.uint8)
        l_r = u_l + np.array([block_H, block_W])
        l_r = l_r.astype(np.uint32)

        descriptors_list.append(generate_descriptor(img, bin_num, b_h, b_w, block_H, block_W, u_l, l_r, label))

    np.savetxt(out_name, descriptors_list, delimiter='\t', fmt='%1.2f')
    return descriptors_list


def load_file(img_dir, args):
    input_dir = 'hw5_data/' + args + img_dir
    images = []
    files = []
    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))
        if img is not None:
            images.append(img)
            files.append(filename)
    out = input_dir + '/descriptor-' + img_dir + '.txt'

    return images, out


# descriptors for the train set
grass_img, grass_out = load_file('grass', 'train/')
grass_desc = desc_vectors(grass_img, grass_out, 1)
ocean_img, ocean_out = load_file('ocean', 'train/')
ocean_desc = desc_vectors(ocean_img, ocean_out, 2)
redcarpet_img, redcarpet_out = load_file('redcarpet', 'train/')
redcarpet_desc = desc_vectors(redcarpet_img, redcarpet_out, 3)
road_img, road_out = load_file('road', 'train/')
road_desc = desc_vectors(road_img, road_out, 4)
wheatfield_img, wheatfield_out = load_file('wheatfield', 'train/')
wheatfield_desc = desc_vectors(wheatfield_img, wheatfield_out, 5)


descriptors = np.vstack((grass_desc, ocean_desc, redcarpet_desc, road_desc, wheatfield_desc))
with open("train.pkl", "wb") as f:
    pickle.dump(descriptors, f)

# descriptors for test set
grass_img, grass_out = load_file('grass', 'test/')
grass_desc = desc_vectors(grass_img, grass_out, 1)
ocean_img, ocean_out = load_file('ocean', 'test/')
ocean_desc = desc_vectors(ocean_img, ocean_out, 2)
redcarpet_img, redcarpet_out = load_file('redcarpet', 'test/')
redcarpet_desc = desc_vectors(redcarpet_img, redcarpet_out, 3)
road_img, road_out = load_file('road', 'test/')
road_desc = desc_vectors(road_img, road_out, 4)
wheatfield_img, wheatfield_out = load_file('wheatfield', 'test/')
wheatfield_desc = desc_vectors(wheatfield_img, wheatfield_out, 5)


descriptors = np.vstack((grass_desc, ocean_desc, redcarpet_desc, road_desc, wheatfield_desc))
with open("test.pkl", "wb") as f:
    pickle.dump(descriptors, f)
