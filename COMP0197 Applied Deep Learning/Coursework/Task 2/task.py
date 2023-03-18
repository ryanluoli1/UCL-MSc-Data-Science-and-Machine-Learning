import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


class mixup:

    def __init__(self, sampling_method, **params):
        # super(mixup, self).__init__()
        self.sampling_method = sampling_method
        self.alpha = params['alpha']
        self.low = params['low']
        self.high = params['high']

    def transform(self, x, y):
        if self.sampling_method == 1:
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1
        elif self.sampling_method == 2:
            lam = np.random.uniform(self.low, self.high)
        batch_size = x.shape[0]
        index = torch.randperm(batch_size)
        x_mixed = lam * x + (1-lam) * x[index, :]
        y_a, y_b = y, y[index]
        return x_mixed, y_a, y_b, lam


class ResNet50(nn.Module):

    def __init__(self, num_classes, pre_trained=False):
        super(ResNet50, self).__init__()
        if pre_trained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.resnet = models.resnet50()
        n_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_in, num_classes)

    def forward(self, x):
        y = self.resnet(x)
        return y


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def train(model, trainloader, testloader, mixer, num_epochs, learning_rate, device, report_loss=True):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        running_loss, batch, total = 0, 0, 0
        for x, y in trainloader:
            total += y.shape[0]
            x, y = x.to(device), y.to(device)
            mixed_x, y_a, y_b, lam = mixer.transform(x, y)
            preds = model(mixed_x)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            running_loss += loss.item()
            batch += 1
            if report_loss and batch % 100 == 0:
                print("Epoch {}, Batch: {}, Running Loss {:.4f}".format(epoch+1, batch, running_loss/total))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = running_loss / (len(trainloader) * trainloader.batch_size)
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                predictions = model(x)
                _, y_pred = torch.max(predictions.data, 1)
                correct += (y_pred==y).sum().item()
        accuracy = 100 * correct / len(testloader.dataset)
        if report_loss:
            print("Epoch {}, Training Loss: {:.4f}, Test Accuracy {:.2f}%".format(epoch+1, train_loss, accuracy))
    print('Training done!')


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    elif torch.has_mps:
        device = torch.device("mps")
        print("M1 chip is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # load the CIFAR10 dataset, reshape images into 3x224x224
    batch_size, test_batch_size = 32, 128
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainset = data.Subset(trainset, np.arange(200))
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testset = data.Subset(testset, np.arange(200))
    testloader = data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)

    # example images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # apply mixup data augmentation on example images
    mixup_params = {'alpha':1.,
                    'low':0,
                    'high':1}
    mixer = mixup(1, **mixup_params)
    mixed_x, y_a, y_b, lam = mixer.transform(images, labels)

    # visualize 16 mixed images
    mixed_images = Image.fromarray((torch.cat(mixed_x[:16].split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    mixed_images.save("mixup.png")
    print('Mixed images saved.')

    # train the ResNet50 models with beta distribution sampling
    num_epochs = 10
    learning_rate = 0.001
    mixer_1 = mixup(1, **mixup_params)
    model_1 = ResNet50(num_classes, pre_trained=False)
    model_1.to(device)
    train(model_1, trainloader, testloader, mixer_1, num_epochs, learning_rate, device)
    torch.save(model_1.state_dict(), 'model_1.pt')
    print('Model 1 saved!')

    # train the ResNet50 models with uniform distribution sampling
    mixer_2 = mixup(2, **mixup_params)
    model_2 = ResNet50(num_classes, pre_trained=False)
    model_2.to(device)
    train(model_2, trainloader, testloader, mixer_2, num_epochs, learning_rate, device)
    torch.save(model_2.state_dict(), 'model_2.pt')
    print('Model 2 saved!')

    # make predictions on 36 test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images[:36]
    test_images = Image.fromarray((torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    test_images.save("result.png")
    print('Test images saved.')
    images = images.to(device)
    model_1.eval()
    preds_1 = model_1(images)
    _, y_pred_1 = torch.max(preds_1.data, 1)
    model_2.eval()
    preds_2 = model_2(images)
    _, y_pred_2 = torch.max(preds_2.data, 1)
    print('Ground-truth Labels : ', [classes[labels[j]] for j in range(36)])
    print('Model 1 Predictions : ', [classes[y_pred_1[j]] for j in range(36)])
    print('Model 2 Predictions : ', [classes[y_pred_2[j]] for j in range(36)])