import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from vars.data_dl import data_loader


def load_sam():
    sam_checkpoint = "./saved_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamAutomaticMaskGenerator(sam)


def cifar10():
    pass


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # Change the output size to 2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initilize(self):

        net = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def train(self):

        trainloader, testloader, classes = data_loader()
        # For Training purpose only
        num_epochs = 10

        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print(
                        f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            # Evaluate the model on the test set after each epoch
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Epoch {epoch+1} Accuracy: {accuracy:.2f}%')

        print('Finished Training')
