'''
Skeleton model class. You will have to implement the classification and regression layers, along with the forward definition.
'''

import datasets
import cv2
import sys
import time
import evaluation
import torch
import numpy as np

from torchvision import models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((224, 224)),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class RCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(RCNN, self).__init__()

        # Pretrained backbone. You may experiment with other models: https://pytorch.org/vision/stable/models.html
        resnet = models.resnet50(pretrained=True)

        # Remove the last two layers of the pretrained network. (This may differ if not using ResNet)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # TODO: Implement the fully connected layers for classification and regression.
        self.fc1 = nn.Linear(in_features=2048 * 7 * 7, out_features=num_classes + 1)
        self.fc2 = nn.Linear(in_features=2048 * 7 * 7, out_features=num_classes * 4)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        # Freeze backbone weights. 
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # TODO: Implement forward. Should return a (batch_size x num_classes) tensor for classification
        #           and a (batch_size x num_classes x 4) tensor for the bounding box regression.
        backbone_x = self.backbone(x)
        backbone_x = backbone_x.view(-1, 2048 * 7 * 7)

        classification = self.dropout1(backbone_x)
        classification = self.fc1(classification)

        regression = self.dropout1(backbone_x)
        regression = self.fc2(regression)

        return classification  # , regression


def RUN(model, data_loader, device, optimizer, loss_fn, data_size,
        train=False, validation=False, test=False):

    if train or validation:
        if train:
            model.train(True)
        if validation:
            with torch.no_grad():
                model.eval()
            y_pred = [0] * _num_classes
            y_true = [0] * _num_classes

        loss_sum = 0.0
        accuracy_sum = 0.0

        for i, item in enumerate(data_loader):
            candidate_image, candidate_region, ground_truth_regions, ground_truth_classes = item
            candidate_image = candidate_image.to(device)
            ground_truth_classes = ground_truth_classes.to(device)

            # predict classes
            outputs = model(candidate_image).to(device)
            # print(outputs.data)

            # compute loss
            loss = loss_fn(outputs, ground_truth_classes)

            optimizer.zero_grad()

            # backpropagate the loss
            loss.backward()

            optimizer.step()

            # keep track of accumulated loss or accuracy.
            loss_sum += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            accuracy_sum += torch.sum(prediction == ground_truth_classes)

            '''
            if validation:
                y_true = y_true[np.nditer(np.array(ground_truth_classes.data[0]))] + 1
                y_pred = y_pred[bp.nditer(np.array(prediction.data[0]))] + 1

        if validation:
            ax = plot_confusion_matrix(y_true, y_pred)
            ax.savefig('cm.png')
        '''

    elif test:

        with torch.no_grad():
            model.eval()

            loss_sum = 0.0
            accuracy_sum = 0.0

            for i, item in enumerate(data_loader):
                img, candidate_image, candidate_region, ground_truth_regions, ground_truth_classes = item
                # print(candidate_image.shape)
                # for coords in candidate_region:
                # print('coords', coords.shape)
                # _candidate_image = transformer(candidate_image[coords[0]:coords[2], coords[1]:coords[3], :])

                # print("candidate img", _candidate_image.shape)
                candidate_image = candidate_image.to(device)
                ground_truth_classes = ground_truth_classes.to(device)

                outputs = model(candidate_image).to(device)
                # print("valid, test output", outputs.shape)

                # compute loss
                loss = loss_fn(outputs, ground_truth_classes)

                # compute loss and accuracy
                loss_sum += loss.item()
                _, prediction = torch.max(outputs.data, 1)
                accuracy_sum += torch.sum(prediction == ground_truth_classes.data)

                if test:
                    c = (prediction == ground_truth_classes).squeeze()
                    for i in range(_num_classes + 1):
                        label = ground_truth_classes[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

    return accuracy_sum / data_size, loss_sum / data_size, optimizer


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax


def collate_fn(batch):
    candidate_images, ground_truth_classes = [], []
    # print(f"batch: {batch}")
    for item in batch:
        i = 0
        for a in item:
            if i != 0:
                print(i)
                print(a.shape)
            i += 1
            # candidate_images = transforms(candidate_image[ground_truth_regions[0]:ground_truth_regions[2], ground_truth_regions[1]:ground_truth_regions[3], :])
            # ground_truth_classes.append(ground_truth_class)
    return torch.Tensor(candidate_images), torch.LongTensor(ground_truth_classes)


if __name__ == '__main__':
    device = 'cuda'

    data_train = datasets.HW6Dataset('./hw6_data_small/train', './hw6_data_small/train.json', candidate_image_size=224)
    data_valid = datasets.HW6Dataset('./hw6_data_small/valid', './hw6_data_small/valid.json', candidate_image_size=224)
    data_test = datasets.HW6DatasetTest('./hw6_data_small/test', './hw6_data_small/test.json', candidate_image_size=224)

    '''
    data_train = datasets.HW6Dataset('./hw6_data_full/train', './hw6_data_full/train.json', candidate_image_size=224)
    data_valid = datasets.HW6Dataset('./hw6_data_full/valid', './hw6_data_full/valid.json', candidate_image_size=224)
    data_test = datasets.HW6DatasetTest('./hw6_data_full/test', './hw6_data_full/test.json', candidate_image_size=224)
    '''

    _num_classes = 4  # 20
    num_epochs = 20
    mini_batch = 8

    # batches using dataloader
    train_dataloader = DataLoader(data_train, batch_size=mini_batch, shuffle=True)
    valid_dataloader = DataLoader(data_valid, batch_size=mini_batch, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=1, shuffle=True)
    # test_dataloader = DataLoader(data_test, batch_size=1, shuffle=True, collate_fn=collate_fn)

    network = RCNN(num_classes=_num_classes).to(device)

    # Adam optimizer and loss function
    optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = nn.CrossEntropyLoss()
    # loss_regression = nn.MSELoss()

    # loss and accuracy container
    training_accuracy_list = []
    training_loss_list = []
    validation_accuracy_list = []
    validation_loss_list = []

    global_v_loss = 10.0
    global_v_acc = 0.0
    global_tr_loss = 10.0
    global_tr_acc = 0.0


    # ------------------------------------------------------------------------------------------------------------------
    print('batch size: %d\n' % mini_batch)
    print(' Epoch     Training accuracy        Training loss       Validation accuracy       Validation loss')

    for epoch in range(num_epochs):

        # training
        training_accuracy, training_loss, _ = RUN(network, data_loader=train_dataloader, device=device,
                                                  optimizer=optimizer, loss_fn=loss, data_size=data_train.__len__(),
                                                  train=True)
        training_accuracy_list.append(training_accuracy)
        training_loss_list.append(training_loss)

        # validation
        validation_accuracy, validation_loss, optimizer = RUN(network, data_loader=valid_dataloader, device=device,
                                                              optimizer=optimizer, loss_fn=loss,
                                                              data_size=data_valid.__len__(),
                                                              validation=True)
        validation_accuracy_list.append(validation_accuracy)
        validation_loss_list.append(validation_loss)

        # save the best model
        if validation_loss < global_v_loss and validation_accuracy > global_v_acc:
            checkpoint = {
                'state_dict': network.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, 'checkpoint.pth')
            global_v_loss = validation_loss
            global_v_acc = validation_accuracy
            global_tr_loss = training_loss
            global_tr_acc = training_accuracy

            # print status
        print(f'({epoch + 1:02d}/{num_epochs})         {training_accuracy:.4f}                 {training_loss:.4f}  \
                {validation_accuracy:.4f}                  {validation_loss:.4f}')

    # load best model
    checkpoint = torch.load('checkpoint.pth')
    network.load_state_dict(checkpoint['state_dict'])
    for parameter in network.parameters():
        parameter.requires_grad = False

    # perform test
    '''
    test_accuracy, _, _ = RUN(network, data_loader=test_dataloader, device=device,
                              optimizer=optimizer, loss_fn=loss, data_size=data_test.__len__(),
                              test=True)
    '''


    # graph plot
    epoch = [i for i in range(1, num_epochs + 1)]
    plt.figure(0)
    plt.plot(epoch, validation_loss_list, 'darkslateblue', label='Validation Loss')
    plt.plot(epoch, training_loss_list, 'deeppink', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig('loss_4.png')

    epoch = [i for i in range(1, num_epochs + 1)]
    plt.figure(1)
    plt.plot(epoch, validation_accuracy_list, 'darkslateblue', label='Validation Accuracy')
    plt.plot(epoch, training_accuracy_list, 'deeppink', label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.savefig('acc_4.png')

    sys.exit()
