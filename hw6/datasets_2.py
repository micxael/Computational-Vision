'''
Datasets for Train/Valid and Test. You should not need to modify anything in this file.
UPDATED: 4/16 - This should be compatible with older versions of pytorch/torchvision.
'''


import cv2
import json
import torch
import torchvision.transforms as transforms
import numpy as np

from os.path import join
from torch.utils.data import Dataset


class HW6Dataset(Dataset):
    '''
    Dataset for Train and Validation.
    Input:
        data_root - path to either the train or valid image directories
        json_file - path to either train.json or valid.json
        candidate_image_size
    Output:
        candidate_image - 3 x M x M tensor
        candidate_region - 1 x 4 tensor
        ground_truth_region - 1 x 4 tensor
        ground_truth_class
    '''
    def __init__(self, data_root, json_file, candidate_image_size):
        with open(json_file, 'r') as f:
            data_dict = json.load(f)

        self.data_root = data_root

        self.images = []
        self.candidate_regions = torch.empty((0, 4), dtype=int)
        self.ground_truth_regions = torch.empty((0, 4), dtype=int)
        self.ground_truth_classes = torch.empty(0, dtype=int)
        for key, values in data_dict.items():
            for val in values:
                self.images.append(key)
                self.candidate_regions = torch.cat((self.candidate_regions, torch.tensor(val['bbox']).unsqueeze(0)), dim=0)
                self.ground_truth_regions = torch.cat((self.ground_truth_regions, torch.tensor(val['correct_bbox']).unsqueeze(0)), dim=0)
                self.ground_truth_classes = torch.cat((self.ground_truth_classes, torch.tensor(val['class']).unsqueeze(0)))

        self.candidate_image_size = candidate_image_size

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Transform to convert to tensor, resize, and normalize.
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])

        self.cached_image = None
        self.cached_image_path = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image.
        image_path = join(self.data_root, self.images[idx])
        if self.cached_image_path != image_path:
            image = cv2.imread(image_path)
            self.cached_image_path = image_path
            self.cached_image = image
        else:
            image = self.cached_image

        # Crop the image to the candidate region.
        candidate_region = self.candidate_regions[idx, :]
        candidate_image = image[candidate_region[1]:candidate_region[3], candidate_region[0]:candidate_region[2], :]

        # Resize candidate image.
        candidate_image = cv2.resize(candidate_image, (self.candidate_image_size, self.candidate_image_size))

        # Apply transform to resize and normalize the candidate image.
        candidate_image = self.transform(candidate_image)

        return candidate_image, candidate_region, self.ground_truth_regions[idx, :], self.ground_truth_classes[idx]


class HW6DatasetTest(Dataset):
    """
    Dataset for Test.
    Input:
        data_root - path to the test image directory
        json_file - path to test.json
        candidate_image_size
    Returns:
        image - numpy array (BGR)
        candidate_images - NUM_CANDIDATE_REGIONS x 3 x M x M tensor
        candidate_regions - all candidate regions for an image 
        ground_truth_regions - all ground truth regions for an image
        ground_truth_classes - all ground truth classes for an image
    """
    def __init__(self, data_root, json_file, candidate_image_size):
        with open(json_file, 'r') as f:
            data_dict = json.load(f)

        self.data_root = data_root
        #print(data_root, flush=True)
        self.images = []
        self.candidate_regions = []
        self.ground_truth_regions = []
        self.ground_truth_classes = []
        for key, values in data_dict.items():
            self.images.append(key)

            regions = torch.empty((len(values['candidate_regions']), 4), dtype=int)
            for i, region in enumerate(values['candidate_regions']):
                regions[i, :] = torch.tensor(region)
            self.candidate_regions.append(regions)

            labels = torch.empty((len(values['ground_truth_regions'])), dtype=int)
            regions = torch.empty((len(values['ground_truth_regions']), 4), dtype=int)
            for i, region in enumerate(values['ground_truth_regions']):
                regions[i, :] = torch.tensor(region['bbox'])
                labels[i] = region['class']
            self.ground_truth_regions.append(regions)
            self.ground_truth_classes.append(labels)

        self.candidate_image_size = candidate_image_size

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Transform to convert to tensor, resize, and normalize.
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image.
        #print("Hi", flush=True)

        image_path = join(self.data_root, self.images[idx])
        #print(image_path, flush=True)
        image = cv2.imread(image_path)
        #print(image)
        # Apply transform to resize and normalize the candidate images.
        idx_candidate_regions = self.candidate_regions[idx]
        candidate_images = torch.empty((len(idx_candidate_regions), 3, self.candidate_image_size, self.candidate_image_size))
        for i, region in enumerate(idx_candidate_regions):
            candidate_image = image[region[1]:region[3], region[0]:region[2], :]
            candidate_image = cv2.resize(candidate_image, (self.candidate_image_size, self.candidate_image_size))
            candidate_image = self.transform(candidate_image)
            candidate_images[i] = candidate_image

        return image, candidate_images, self.candidate_regions[idx], self.ground_truth_regions[idx], self.ground_truth_classes[idx]


if __name__ == '__main__':
    # Example usage.
    data_train = HW6Dataset('./hw6_data_small/train/', './hw6_data_small/train.json', candidate_image_size=224)
    data_test = HW6DatasetTest('./hw6_data_small/test/', './hw6_data_small/test.json', candidate_image_size=224)
    x, y, z, k = data_train.__getitem__(14)
    #print(y[0].shape)
    # print(z)

    cv2.imshow('', np.array(x[0]))
    cv2.waitKey(0)

    print(y)
    print(z)
    print(k)

