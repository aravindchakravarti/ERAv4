"""
Utilities for dataset, dataloaders, and transforms
"""

import torch
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
from constants import dataset_constants

def check_cuda_available():
    """Return True if CUDA GPU is available, else False."""
    return torch.cuda.is_available()


def train_dataset_transformer(**kwargs):
    """Builds a transform pipeline for training dataset from keyword arguments."""
    transform_list = []
    mnist_mean = dataset_constants.dataset_mean
    mnist_std = dataset_constants.dataset_std

    for arg, value in kwargs.items():
        if arg == 'CenterCrop':
            transform_list.append(transforms.CenterCrop(value))

        elif arg == 'Resize':
            transform_list.append(transforms.Resize(value))

        elif arg == 'RandomRotation':
            degrees = value.get('degrees', (-15., 15.))
            fill = value.get('fill', 0)
            transform_list.append(transforms.RandomRotation(degrees, fill=fill))

        elif arg == 'RandomApply':
            transform_list.append(
                transforms.RandomApply([value['transform']], p=value.get('p', 0.5))
            )

        elif arg == 'ToTensor' and value:
            transform_list.append(transforms.ToTensor())

        elif arg == 'Normalize':
            mean = value.get('mean', mnist_mean)
            std = value.get('std', mnist_std)
            transform_list.append(transforms.Normalize(mean, std))

        else:
            raise ValueError(f"Unknown transform: {arg}")

    return transforms.Compose(transform_list)


def test_dataset_transformer(**kwargs):
    """Builds a transform pipeline for test dataset from keyword arguments."""
    transform_list = []
    mnist_mean = dataset_constants.dataset_mean
    mnist_std = dataset_constants.dataset_std

    for arg, value in kwargs.items():
        if arg == 'ToTensor' and value:
            transform_list.append(transforms.ToTensor())
        elif arg == 'Normalize':
            mean = value.get('mean', mnist_mean)
            std = value.get('std', mnist_std)
            transform_list.append(transforms.Normalize(mean, std))
        else:
            raise ValueError(f"Unknown transform: {arg}")
    return transforms.Compose(transform_list)


def get_dataset(path, train_transforms, test_transforms):
    """Return MNIST train and test datasets with given transform."""
    train_data = datasets.MNIST(path, train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST(path, train=False, download=True, transform=test_transforms)
    return train_data, test_data


def get_dataset_loaders(train_data, test_data, batch_size, use_cuda):
    """Return DataLoaders for train and test datasets."""
    train_kwargs = {'batch_size': batch_size, 'shuffle': True,  'num_workers': 2, 'pin_memory': use_cuda}
    test_kwargs  = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': use_cuda}

    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data,  **test_kwargs)
    return train_loader, test_loader


def dataset_visualizer(dataset_loader, n_images=12):
    """Visualize a few samples from the dataset loader."""
    batch_data, batch_label = next(iter(dataset_loader))
    n_images = min(n_images, len(batch_data))

    fig = plt.figure(figsize=(10, 8))
    for i in range(n_images):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
    plt.show()


''' Usage: 
train_transforms = train_dataset_transformer( 
    RandomApply={'transform': transforms.CenterCrop(4), 'p': 0.1}, 
    Resize=(28, 28), 
    RandomRotation={'degrees': (-15., 15.), 'fill': 0}, 
    ToTensor=True, 
    Normalize={'mean': mnist_mean, 'std': mnist_std}
    ) 
'''