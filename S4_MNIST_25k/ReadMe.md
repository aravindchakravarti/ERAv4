# MNIST Image Classification with PyTorch

This project demonstrates a Convolutional Neural Network (CNN) implemented in PyTorch for classifying handwritten digits from the MNIST dataset. The notebook covers data loading, preprocessing, model definition, training, and evaluation.

## Table of Contents
- [Setup](#setup)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Setup

The following Python libraries are required to run this notebook:
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `torch.optim`
- `torchvision.datasets`
- `torchvision.transforms`
- `matplotlib.pyplot`
- `tqdm`

The code automatically checks for CUDA availability and utilizes a GPU if available; otherwise, it defaults to the CPU.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
device = torch.device("cuda" if cuda else "cpu")
```

## Dataset and Preprocessing

The MNIST dataset is downloaded and loaded using `torchvision.datasets.MNIST`. Both training and testing datasets undergo specific transformations:

-   **Training Transformations**:
    -   `transforms.RandomApply([transforms.CenterCrop(22)], p=0.1)`: Randomly applies a center crop to 22x22 with a probability of 0.1.
    -   `transforms.Resize((28, 28))`: Resizes images to 28x28 pixels.
    -   `transforms.RandomRotation((-15., 15.), fill=0)`: Randomly rotates images by -15 to +15 degrees, filling empty areas with 0.
    -   `transforms.ToTensor()`: Converts images to PyTorch tensors.
    -   `transforms.Normalize(mnist_mean, mnist_std)`: Normalizes the pixel values using the calculated mean (0.1307) and standard deviation (0.3081) of the MNIST dataset.

-   **Test Transformations**:
    -   `transforms.ToTensor()`: Converts images to PyTorch tensors.
    -   `transforms.Normalize(mnist_mean, mnist_std)`: Normalizes the pixel values.

Data loaders are set up with a `batch_size` of 128, shuffling for training data, and optimized for CUDA if available (`num_workers=2`, `pin_memory=True`).

## Model Architecture

The CNN model, named `Net`, consists of several convolutional blocks followed by activation functions and pooling layers.

The architecture includes:
-   Multiple `Conv2d` layers with `kernel_size=3` and `padding='same'`.
-   `F.relu` as the activation function after most convolutional layers.
-   `F.max_pool2d` for downsampling.
-   A `Conv2d` layer with `kernel_size=1` for dimensionality reduction within blocks.
-   `F.adaptive_avg_pool2d` to reduce the spatial dimensions to 1x1 before the final output.
-   `torch.flatten` to convert the feature maps into a 1D vector for the final classification layer.

The model has **13,352 trainable parameters**.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(1, 16, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
        self.conv14 = nn.Conv2d(16, 8, kernel_size=1, padding='same')

        self.conv21 = nn.Conv2d(8, 16, kernel_size=3, padding='same')
        self.conv22 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
        self.conv24 = nn.Conv2d(16, 8, kernel_size=1, padding='same')

        self.conv31 = nn.Conv2d(8, 16, kernel_size=3, padding='same')
        self.conv32 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
        self.conv34 = nn.Conv2d(16, 8, kernel_size=1, padding='same')

        self.conv41 = nn.Conv2d(8, 16, kernel_size=3, padding='same')
        self.conv42 = nn.Conv2d(16, 16, kernel_size=3, padding='same')


    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(F.max_pool2d(self.conv14(x), 2))

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(F.max_pool2d(self.conv24(x), 2))

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(F.max_pool2d(self.conv34(x), 2))

        x = F.relu(self.conv41(x))
        x = self.conv42(x)                  # [batch, 10, H, W]
        x = F.adaptive_avg_pool2d(x, (1,1)) # [batch, 10, 1, 1]
        x = torch.flatten(x, 1)             # [batch, 10]
        return x
```

## Training and Evaluation

The training process involves:
-   **Optimizer**: `optim.Adam` with a learning rate of `1e-3`.
-   **Loss Function**: `nn.CrossEntropyLoss`.
-   **Epochs**: The model is trained for 1 epoch.

The `train` function iterates through the training data, performs forward and backward passes, updates model weights, and logs training loss and accuracy. The `test` function evaluates the model on the test set, calculates the test loss, and accuracy.

```python
# Function to get the count of correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

# Training function
def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    pred = model(data)
    loss = criterion(pred, target)
    train_loss+=loss.item()

    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

# Testing function
def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()

            correct += GetCorrectPredCount(output, target)


            test_loss /= len(test_loader)

            test_acc.append(100. * correct / len(test_loader.dataset))
            test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

## Results

After 1 epoch, the model achieved an accuracy of **88.85%** on the test set.

The notebook also generates plots to visualize the training loss, training accuracy, test loss, and test accuracy over the epoch.