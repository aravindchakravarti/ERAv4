import torch
import torch.nn as nn
import torch.nn.functional as F

# Depthwise Separable Convolution Block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 🔹 First block (unchanged)
        BLK_1_CH = 32
        BLK_2_CH = 64
        BLK_3_CH = 128

        self.conv11 = nn.Conv2d(in_channels=3, out_channels=BLK_1_CH, kernel_size=3, padding=1)     
        self.batch11 = nn.BatchNorm2d(BLK_1_CH)
        self.conv12 = nn.Conv2d(in_channels=BLK_1_CH, out_channels=BLK_1_CH, kernel_size=3, padding=1)
        self.batch12 = nn.BatchNorm2d(BLK_1_CH)
        self.conv13 = nn.Conv2d(in_channels=BLK_1_CH, out_channels=BLK_1_CH, kernel_size=3, padding=1)
        self.batch13 = nn.BatchNorm2d(BLK_1_CH)
        self.conv14 = nn.Conv2d(in_channels=BLK_1_CH, out_channels=BLK_2_CH, kernel_size=1)
        self.batch14 = nn.BatchNorm2d(BLK_2_CH)
        self.pool14 = nn.MaxPool2d(2, 2)          
        
        # self.conv21 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)   
        # self.batch21 = nn.BatchNorm2d(32)
        # self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.batch22 = nn.BatchNorm2d(32)
        # self.conv23 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.batch23 = nn.BatchNorm2d(32)
        # self.conv24 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        # self.batch24 = nn.BatchNorm2d(64)
        # self.pool24 = nn.MaxPool2d(2, 2)                                            

        # 🔹 Second block (Depthwise Separable)
        self.conv21 = DepthwiseSeparableConv(BLK_2_CH, BLK_2_CH)
        self.batch21 = nn.BatchNorm2d(BLK_2_CH)
        self.conv22 = DepthwiseSeparableConv(BLK_2_CH, BLK_2_CH)
        self.batch22 = nn.BatchNorm2d(BLK_2_CH)
        self.conv23 = DepthwiseSeparableConv(BLK_2_CH, BLK_2_CH)
        self.batch23 = nn.BatchNorm2d(BLK_2_CH)
        self.conv24 = nn.Conv2d(in_channels=BLK_2_CH, out_channels=BLK_3_CH, kernel_size=1)  # keep 1x1 as-is
        self.batch24 = nn.BatchNorm2d(BLK_3_CH)
        self.pool24 = nn.MaxPool2d(2, 2)                                                     

        # 🔹 Third block (Depthwise Separable)
        self.conv31 = DepthwiseSeparableConv(BLK_3_CH, BLK_3_CH)
        self.batch31 = nn.BatchNorm2d(BLK_3_CH)
        self.conv32 = DepthwiseSeparableConv(BLK_3_CH, BLK_3_CH)
        self.batch32 = nn.BatchNorm2d(BLK_3_CH)
        self.conv33 = DepthwiseSeparableConv(BLK_3_CH, BLK_3_CH)
        self.batch33 = nn.BatchNorm2d(BLK_3_CH)
        self.conv34 = nn.Conv2d(in_channels=BLK_3_CH, out_channels=BLK_3_CH, kernel_size=1)  # keep 1x1 as-is
        self.batch34 = nn.BatchNorm2d(BLK_3_CH)
        self.pool34 = nn.MaxPool2d(2, 2)

        # 🔹 Final classifier
        self.conv_last = nn.Conv2d(in_channels=BLK_3_CH, out_channels=10, kernel_size=1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.batch11(self.conv11(x)))
        x = F.relu(self.batch12(self.conv12(x)))
        x = F.relu(self.batch13(self.conv13(x)))
        x = F.relu(self.batch14(self.conv14(x)))
        x = self.pool14(x)

        # Block 2
        x = F.relu(self.batch21(self.conv21(x)))
        x = F.relu(self.batch22(self.conv22(x)))
        x = F.relu(self.batch23(self.conv23(x)))
        x = F.relu(self.batch24(self.conv24(x)))
        x = self.pool24(x)

        # Block 3
        x = F.relu(self.batch31(self.conv31(x)))
        x = F.relu(self.batch32(self.conv32(x)))
        x = F.relu(self.batch33(self.conv33(x)))
        x = F.relu(self.batch34(self.conv34(x)))
        x = self.pool34(x)

        # Final layers
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x,1)
        return x
