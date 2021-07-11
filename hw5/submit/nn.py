import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler




'''
Fully-connected Neural Net
'''
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()

        # hidden layers sizes, you can play with it as you wish!
        hidden1 = 512
        hidden2 = 256
        hidden3 = 128

        # layer parameters
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.output = nn.Linear(hidden3, num_classes)  # 5

        # activations
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.output(output)

        return output


'''
Convolutional NN
'''
class Net(nn.Module):
    # one Convolution layer
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3,
                              out_channels=out_channels, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)

        return output


class CNN(nn.Module):
    # generate the Convolutional Network
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()

        # Create __ layers of the unit with max pooling in between
        self.layer1 = Net(in_channels=3, out_channels=32)
        self.layer2 = Net(in_channels=32, out_channels=64)
        self.layer3 = Net(in_channels=64, out_channels=128)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropoutFC = nn.Dropout(p=0.5)
        self.dropoutCONV = nn.Dropout(p=0.5)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.layer1,
                                 self.maxpool,
                                 self.layer2,
                                 self.maxpool,
                                 self.layer3,
                                 self.maxpool)

        # dense
        self.fc1 = nn.Linear(in_features=128 * 7 * 11, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=5)

    def forward(self, x):
        output = self.net(x)
        output = output.view(-1, 128 * 7 * 11)
        output = self.dropoutCONV(output)
        output = self.fc1(output)
        output = self.dropoutFC(output)
        output = self.fc2(output)
        return output



def Run(network, data_loader, device, optimizer, loss_fn, data_size, input_shape=None, train=False, test=False):
    # runs the network
    loss_sum = 0.0
    accuracy_sum = 0.0

    # for each data batch
    for i, (images, labels) in enumerate(data_loader):

        if train:
            network.train(True)
        else:  # validation and test
            torch.no_grad()  # memory efficienty ?
            network.eval()

        # load data
        if input_shape is not None:
            images = images.reshape(-1, input_shape).to(device)
        else:
            images = images.to(device)
        labels = labels.to(device)

        outputs = network(images).to(device)

        # compute loss
        loss = loss_fn(outputs, labels)

        if train:
            # clear all accumulated gradients
            optimizer.zero_grad()
            # backpropagate the loss
            loss.backward()
            # adjust parameters
            optimizer.step()

        # store data
        loss_sum += loss.item()
        _, prediction = torch.max(outputs.data, 1)
        accuracy_sum += torch.sum(prediction == labels.data)

        # store test data
        if test:
            c = (prediction == labels).squeeze()
            for j in range(5):
                label = labels[j]
                correct_list[label] += c[j].item()
                total_list[label] += 1

    return accuracy_sum / data_size, loss_sum / data_size, optimizer


def data_loader(input_min_dim, batch_size):
    # image transforming
    data_transforms = transforms.Compose([
        transforms.Resize(input_min_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root="/home/jae/rpi/spring2021/comp_vision/hw5/hw5_data/train", transform=data_transforms)
    test_dataset = datasets.ImageFolder(root="/home/jae/rpi/spring2021/comp_vision/hw5/hw5_data/test", transform=data_transforms)

    # split into training and validation
    percentage = 0.875
    dataset_size = len(dataset) - 7
    train_size = int(dataset_size * percentage)
    valid_size = dataset_size - train_size

    # pseudo-random dataset (change seeds in main)
    indices = torch.randperm(len(dataset))
    train_indices = indices[:len(indices) - valid_size][:train_size or None]
    valid_indices = indices[len(indices) - valid_size:] if valid_size else None

    # load data
    # num_workers: parallel data transfer
    # pin_memory: CPU <-> GPU
    data_loader = DataLoader(dataset, batch_size=dataset_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    train_loader = DataLoader(dataset, pin_memory=True, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=4)
    valid_loader = DataLoader(dataset, pin_memory=True, batch_size=batch_size,
                              sampler=SubsetRandomSampler(valid_indices),
                              num_workers=4)
    test_loader = DataLoader(test_dataset, pin_memory=True, batch_size=batch_size,
                             shuffle=True, num_workers=4)

    return train_loader, valid_loader, test_loader


def plot_data(num_epochs, train_accuracy, train_loss, valid_accuracy, valid_loss):
    # for drawing graphs
    epoch = [i for i in range(num_epochs)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epoch, valid_accuracy, 'darkslateblue', label='Validation Accuracy')
    ax1.plot(epoch, train_accuracy, 'deeppink', label='Training Accuracy')
    ax1.legend()
    ax1.set(xlabel='Epoch')

    ax2.plot(epoch, valid_loss, 'darkslateblue', label='Validation Loss')
    ax2.plot(epoch, train_loss, 'deeppink', label='Training Loss')
    ax2.set(xlabel='Epoch')
    ax2.legend()

    plt.show()


if __name__ == "__main__":

    # set seed value
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    device = 'cuda'  # or cpu

    # test with different variables
    # hyperparameters
    num_classes = 5  # The number of output (image) classes.
    num_epochs = 50
    batch_size = 64


    # model_name and shape
    model_name = sys.argv[1]

    if model_name == 'NN':
        input_shape = 3 * 60 * 90
        network = NN(input_shape, num_classes).to(device)
    elif model_name == 'CNN':
        input_shape = None
        network = CNN(num_classes=5).to(device)
    else:  # wrong input
        print("Usage: python p2.py 'NN or CNN'")
        sys.exit()


    # load data
    train_loader, valid_loader, test_loader = data_loader(input_min_dim=60, batch_size=batch_size)

    # Adam optimizer and loss function
    optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = nn.CrossEntropyLoss()

    # temp holders
    train_loss, train_accuracy, valid_loss, valid_accuracy = [], [], [], []
    best_train_loss = 10.0
    best_train_acc = 0.0
    best_valid_loss = 10.0
    best_valid_acc = 0.0
    correct_list = list(0. for i in range(5))
    total_list = list(0. for i in range(5))
    best_network = None


    # train and validation
    # ------------------------------------------------------------------------------------------------------------------
    print('batch size: %d\n' % batch_size)
    print(' Epoch     Training accuracy        Training loss       Validation accuracy       Validation loss')

    for epoch in range(num_epochs):

        # training
        training_accuracy, training_loss, _ = Run(network, data_loader=train_loader,
                                                  device=device, optimizer=optimizer,
                                                  loss_fn=loss, data_size=3500,
                                                  input_shape=input_shape, train=True)
        train_accuracy.append(training_accuracy)
        train_loss.append(training_loss)

        # validating
        validation_accuracy, validation_loss, optimizer = Run(network, data_loader=valid_loader,
                                                              device=device, optimizer=optimizer,
                                                              loss_fn=loss, data_size=500,
                                                              input_shape=input_shape)
        valid_accuracy.append(validation_accuracy)
        valid_loss.append(validation_loss)

        # save best model
        if validation_loss < best_valid_loss and validation_accuracy > best_valid_acc:
            best_network = {
                'state_dict': network.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            best_train_loss = training_loss
            best_train_acc = training_accuracy
            best_valid_loss = validation_loss
            best_valid_acc = validation_accuracy

        # print status
        print(f'({epoch + 1:02d}/{num_epochs})         {training_accuracy:.4f}                 {training_loss:.4f}  \
                {validation_accuracy:.4f}                  {validation_loss:.4f}')


    # testing
    # ------------------------------------------------------------------------------------------------------------------
    network.load_state_dict(best_network['state_dict'])  # using the best model from learning
    for parameter in network.parameters():
        parameter.requires_grad = False

    # perform test
    test_accuracy, _, _ = Run(network, data_loader=test_loader, device=device,
                              optimizer=optimizer, loss_fn=loss,
                              data_size=1000, input_shape=input_shape,
                              test=True)



    # print info
    plot_data(num_epochs, train_accuracy, train_loss, valid_accuracy, valid_loss)

    print(f'\n Best training loss: {best_train_loss:.6f}\
         \n Best training accuracy: {best_train_acc:.6f}\
         \n Best validation accuracy: {best_valid_acc:.6f}\
         \n Best validation loss: {best_valid_loss:.6f}\
         \n Test accuracy: {test_accuracy:.6f}\n')

    classes = ['Grass', 'Ocean', 'Redcarpet', 'Road', 'Wheatfield']
    for i in range(5):
        print(' Accuracy of class %5s : %.3f%%' % (classes[i], 100 * correct_list[i] / total_list[i]))

    sys.exit()
