import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic Residual Block
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1  # used to calculate output channels in ResNet

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # You can later replace nn.Conv2d with SeparableConv2d
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# -----------------------------
# ResNet Backbone
# -----------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, base_channels=64):
        super(ResNet, self).__init__()

        self.in_channels = base_channels

        # Initial conv layer (optimized for CIFAR: 3x3, stride=1)
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # No maxpool for 32×32 images
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*8*block.expansion, num_classes)

        # Kaiming initialization (recommended for ReLU)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # If input and output dimensions differ, add 1×1 conv shortcut
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -----------------------------
# ResNet Variants
# -----------------------------
def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes=100):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# You can easily create ResNet-21, for example:
def ResNet21(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 3, 2], num_classes=num_classes)
