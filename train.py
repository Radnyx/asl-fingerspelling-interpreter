#  Doron Barasch & RanDair Porter

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from model import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data():

    # Remember that we might need to flip the images horizontally
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=3, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        transforms.RandomPerspective(distortion_scale=.15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])

    data_path = './dataset5/A/'
    dataset = datasets.ImageFolder(data_path, transform=transform_train)
    train_size = int(.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,
                                                num_workers=12)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True,
                                              num_workers=12)

    classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y']
    return {'train': trainloader, 'test': testloader, 'classes': classes}


def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=1):
    net.to(device)
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step() # takes a step in gradient direction

            # print statistics
            losses.append(loss.item())
            sum_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                if verbose:
                  print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    return losses


def accuracy(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total


def smooth(x, size):
    return np.convolve(x, np.ones(size)/size, mode='valid')


if __name__ == '__main__':
    # freeze_support()
    data = get_data()

    print(data['train'].__dict__)
    print(data['test'].__dict__)

    conv_net = ConvNet()

    conv_losses = train(conv_net, data['train'], epochs=15, lr=.01)
    plt.plot(smooth(conv_losses, 50))

    torch.save(conv_net.state_dict(), 'neuralnet7', _use_new_zipfile_serialization=False)

    print("Training accuracy: %f" % accuracy(conv_net, data['train']))
    print("Testing  accuracy: %f" % accuracy(conv_net, data['test']))

