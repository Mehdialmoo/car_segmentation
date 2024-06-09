import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Custom dataset class to filter and relabel the classes


class AutomobileDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.automobile_classes = [1, 9]  # Class indices for 'car' and 'truck'

        # Create a mapping of original labels to new binary labels
        self.label_map = {
            label: 1 if label in self.automobile_classes else 0 for label in range(10)}

        # Filter the dataset to include only automobile
        # and non-automobile samples
        self.indices = [i for i, (_, label) in enumerate(
            dataset) if label in self.automobile_classes or label not in self.automobile_classes]

    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        label = self.label_map[label]
        return img, label

    def __len__(self):
        return len(self.indices)


def download_cifar():
    # download the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    return trainset, testset


def load_cifar():

    trainset, testset = download_cifar()
    # Create the automobile datasets
    trainset_auto = AutomobileDataset(trainset)
    testset_auto = AutomobileDataset(testset)

    return trainset_auto, testset_auto


def data_loader():

    trainset_auto, testset_auto = load_cifar()
    # Create the data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset_auto, batch_size=6, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset_auto, batch_size=6, shuffle=False, num_workers=2)

    classes = ('others', 'automobile')

    return trainloader, testloader, classes


def sample_show():

    trainset_auto, testset_auto = load_cifar()
    # Get a balanced subset of images from the training set
    automobile_indices = [i for i, (_, label) in enumerate(
        trainset_auto) if label == 1]
    other_indices = [i for i, (_, label) in enumerate(
        trainset_auto) if label == 0]

    selected_indices = automobile_indices[:32] + other_indices[:32]
    selected_dataset = torch.utils.data.Subset(trainset_auto, selected_indices)

    # Create a data loader for the selected images
    selected_loader = torch.utils.data.DataLoader(
        selected_dataset, batch_size=64, shuffle=False)

    # Get the selected images and labels
    images, labels = next(iter(selected_loader))

    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images, nrow=8)

    # Show the grid of images
    img_grid = img_grid / 2 + 0.5     # unnormalize
    npimg = img_grid.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # Remove axis
    plt.show()

    # Print the labels
    label_names = ['automobile' if label ==
                   1 else 'others' for label in labels]
    print(' '.join('%5s' % label_names[j] for j in range(64)))
