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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data():

    # Remember that we might need to flip the images horizontally
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    train_data_path = './archive/asl_alphabet_train/asl_alphabet_train/'
    train_dataset = datasets.ImageFolder(train_data_path, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,
                                                num_workers=12)

    test_data_path = './archive/asl_alphabet_test/asl_alphabet_test/'
    test_dataset = datasets.ImageFolder(test_data_path, transform_test)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                             num_workers=12)
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']
    return {'train': trainloader, 'test': testloader, 'classes': classes}


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(4096, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


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

    torch.save(conv_net.state_dict(), './neuralnet')

    print("Training accuracy: %f" % accuracy(conv_net, data['train']))
    print("Testing  accuracy: %f" % accuracy(conv_net, data['test']))

