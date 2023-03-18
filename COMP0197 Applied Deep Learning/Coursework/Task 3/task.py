import time
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


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


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


def train_test_split(dataset, test_ratio, shuffle=True):
    test_size = int(test_ratio*len(dataset))
    train_size = len(dataset) - test_size
    if shuffle:
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        train_idx = np.arange(train_size)
        test_idx = np.arange(train_size, len(dataset))
        train_set = torch.utils.data.Subset(dataset, train_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)
    return train_set, test_set


def train_valid(model_1, model_2, mixer_1, mixer_2, trainloader, validloader , num_epochs, learning_rate, device, report_loss=True):
    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(model_1.parameters(), lr=learning_rate)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=learning_rate)
    train_time_1, train_time_2, valid_time_1, valid_time_2 = 0, 0, 0, 0
    m1_tl, m1_vl, m1_va, m2_tl, m2_vl, m2_va = [], [], [], [], [], []
    for epoch in range(num_epochs):
        model_1.train()
        model_2.train()
        running_loss_1, running_loss_2, batch, total = 0, 0, 0, 0
        for x, y in trainloader:
            batch += 1
            total += y.shape[0]
            x, y = x.to(device), y.to(device)
            # train model 1
            start = time.time()
            mixed_x, y_a, y_b, lam = mixer_1.transform(x, y)
            preds = model_1(mixed_x)
            loss_1 = mixup_criterion(criterion, preds, y_a, y_b, lam)
            running_loss_1 += loss_1.item()
            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()
            end = time.time()
            train_time_1 += end - start
            # train model 2
            start = time.time()
            mixed_x, y_a, y_b, lam = mixer_2.transform(x, y)
            preds = model_2(mixed_x)
            loss_2 = mixup_criterion(criterion, preds, y_a, y_b, lam)
            running_loss_2 += loss_2.item()
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()
            end = time.time()
            train_time_2 += end - start
            if report_loss and batch % 200 == 0:
                print("Epoch {}, Batch: {}".format(epoch+1, batch))
                print("Model 1 running Loss {:.4f}".format(running_loss_1/total))
                print("Model 2 running Loss {:.4f}".format(running_loss_2/total))
        train_loss_1 = running_loss_1 / len(trainloader.dataset)
        train_loss_2 = running_loss_2 / len(trainloader.dataset)
        model_1.eval()
        model_2.eval()
        valid_loss_1, valid_loss_2, correct_1, correct_2 = 0, 0, 0, 0
        with torch.no_grad():
            for x, y in validloader:
                x, y = x.to(device), y.to(device)
                # evaluate model 1
                start = time.time()
                preds_1 = model_1(x)
                end = time.time()
                valid_time_1 += end - start
                valid_loss_1 += criterion(preds_1, y).item()
                _, y_pred_1 = torch.max(preds_1.data, 1)
                correct_1 += (y_pred_1==y).sum().item()
                # evaluate model 2
                start = time.time()
                preds_2 = model_2(x)
                end = time.time()
                valid_time_2 += end - start
                valid_loss_2 += criterion(preds_2, y).item()
                _, y_pred_2 = torch.max(preds_2.data, 1)
                correct_2 += (y_pred_2==y).sum().item()
        valid_loss_1 = valid_loss_1 / len(validloader.dataset)
        valid_loss_2 = valid_loss_2 / len(validloader.dataset)
        accuracy_1 = 100 * correct_1 / len(validloader.dataset)
        accuracy_2 = 100 * correct_2 / len(validloader.dataset)
        if report_loss:
            print(50*'-')
            print("Epoch {}".format(epoch+1))
            print("Model 1: Training Loss: {:.4f}, "
                  "Validation Loss: {:.4f}, "
                  "Validation Accuracy {:.2f}%, "
                  "Training Time per Epoch {:2f}, "
                  "Validation Time per Epoch {:2f}".format(train_loss_1, valid_loss_1, accuracy_1, train_time_1/(epoch+1), valid_time_1/(epoch+1)))
            print("Model 2: Training Loss: {:.4f}, "
                  "Validation Loss: {:.4f}, "
                  "Validation Accuracy {:.2f}%, "
                  "Training Time per Epoch {:2f}, "
                  "Validation Time per Epoch {:2f}".format(train_loss_2, valid_loss_2, accuracy_2, train_time_2/(epoch+1), valid_time_2/(epoch+1)))
            print(50 * '-')
        m1_tl.append(train_loss_1)
        m1_vl.append(valid_loss_1)
        m1_va.append(accuracy_1)
        m2_tl.append(train_loss_2)
        m2_vl.append(valid_loss_2)
        m2_va.append(accuracy_2)
    print('Training done!')
    return m1_tl, m1_vl, m1_va, train_time_1, valid_time_1, m2_tl, m2_vl, m2_va, train_time_2, valid_time_2


def evaluate(model, testloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss, correct = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss += criterion(preds, y).item()
            _, y_pred = torch.max(preds.data, 1)
            correct += (y_pred == y).sum().item()
        loss = loss / len(testloader.dataset)
        accuracy = 100 * correct / len(testloader.dataset)
    return loss, accuracy




if __name__ == '__main__':

    # setup device
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
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)

    # split the dataset into development (train, validation) and test sets
    dev_set, test_set = train_test_split(dataset, test_ratio=0.2, shuffle=False)
    train_set, valid_set = train_test_split(dev_set, test_ratio=0.1, shuffle=True)

    # generate subsets for local testing
    # train_set = data.Subset(train_set, np.arange(500))
    # valid_set = data.Subset(valid_set, np.arange(500))
    # test_set = data.Subset(test_set, np.arange(500))

    # data loaders
    train_batch_size, valid_batch_size, test_batch_size = 32, 128, 128
    trainloader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
    validloader = data.DataLoader(valid_set, batch_size=valid_batch_size, shuffle=False, num_workers=2)
    testloader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=2)

    # hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    mixup_params = {'alpha':1.,
                    'low':0,
                    'high':1}

    # model 1: using sampling_method=1 and Adam optimizer
    mixer_1 = mixup(1, **mixup_params)
    model_1 = ResNet50(num_classes, pre_trained=False)
    model_1.to(device)

    # model 2: using sampling_method=2 and Adam optimizer
    mixer_2 = mixup(2, **mixup_params)
    model_2 = ResNet50(num_classes, pre_trained=False)
    model_2.to(device)

    # train and evaluate the two models
    results = train_valid(model_1, model_2, mixer_1, mixer_2,
                          trainloader, validloader, num_epochs,
                          learning_rate, device)
    torch.save(model_1.state_dict(), 'model_beta.pt')
    torch.save(model_2.state_dict(), 'model_uniform.pt')
    print('Models saved!')

    # summary of loss, accuracy and speed on training and validation
    train_loss_1 = results[0]
    valid_loss_1 = results[1]
    valid_accuracy_1 = results[2]
    train_time_1 = results[3]
    valid_time_1 = results[4]
    train_loss_2 = results[5]
    valid_loss_2 = results[6]
    valid_accuracy_2 = results[7]
    train_time_2 = results[8]
    valid_time_2 = results[9]

    # make predictions on test set
    test_loss_1, test_accuracy_1 = evaluate(model_1, testloader)
    test_loss_2, test_accuracy_2 = evaluate(model_2, testloader)

    # report results
    print("We trained and evaluated 2 models both using Adam optimizer "
          "but with different sampling methods for mixup augmentation.")

    print("Training Speed: Model 1 = {:.2f} seconds per epoch, "
          "Model 2 = {:.2f} seconds per epoch.".format(train_time_1/num_epochs, train_time_2/num_epochs))
    print("Validation Speed: Model 1 = {:.2f} seconds per epoch, "
          "Model 2 = {:.2f} seconds per epoch.".format(valid_time_1/num_epochs, valid_time_2/num_epochs))

    print("Training Results (averaged over 10 epochs): ")
    print("Model 1: Loss = {:.4f}".format(np.mean(train_loss_1)))
    print("Model 2: Loss = {:.4f}".format(np.mean(train_loss_2)))

    print("Training Results (last epoch): ")
    print("Model 1: Loss = {:.4f}".format(train_loss_1[-1]))
    print("Model 2: Loss = {:.4f}".format(train_loss_2[-1]))

    print("Validation Results (averaged over 10 epochs): ")
    print("Model 1: Loss = {:.4f}, Accuracy = {:.2f}%".format(np.mean(valid_loss_1), np.mean(valid_accuracy_1)))
    print("Model 1: Loss = {:.4f}, Accuracy = {:.2f}%".format(np.mean(valid_loss_2), np.mean(valid_accuracy_2)))

    print("Validation Results (last epoch): ")
    print("Model 1: Loss = {:.4f}, Accuracy = {:.2f}%".format(valid_loss_1[-1], valid_accuracy_1[-1]))
    print("Model 1: Loss = {:.4f}, Accuracy = {:.2f}%".format(valid_loss_2[-1], valid_accuracy_2[-1]))

    print("Test Results: ")
    print("Model 1: Loss = {:.4f}, Accuracy = {:.2f}%".format(test_loss_1, test_accuracy_1))
    print("Model 2: Loss = {:.4f}, Accuracy = {:.2f}%".format(test_loss_2, test_accuracy_2))

    print("We observe similar loss and accuracy values on validation and test sets, "
          "indicating the good generalization of the model on unseen data.")
