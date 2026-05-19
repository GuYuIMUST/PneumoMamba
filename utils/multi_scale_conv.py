"""
Multi-scale Dilated Residual Module (MDR).

Extracts multi-scale features using asymmetric convolution decomposition
(1xK + Kx1) with different kernel sizes and dilation rates, combined
with CPCA (Channel-Positional Cross Attention) blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention via parallel average and max pooling."""

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class CPCABlock(nn.Module):
    """Channel-Positional Cross Attention block for the MDR module.

    Combines channel attention with multi-scale depthwise spatial convolutions.
    """

    def __init__(self, in_channels, out_channels, channelAttention_reduce=4):
        super().__init__()
        self.C = in_channels
        self.O = out_channels
        assert in_channels == out_channels

        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


class BNPReLU(nn.Module):
    """BatchNorm followed by PReLU activation."""

    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class MultiScaleConvModule(nn.Module):
    """Multi-scale Dilated Residual (MDR) convolution module.

    Uses three parallel branches with asymmetric convolution decomposition:
    - Branch 1: 1x3 + 3x1 + 3x3(dilation=3)
    - Branch 2: 1x5 + 5x1 + 3x3(dilation=5)
    - Branch 3: 1x7 + 7x1 + 3x3(dilation=7)

    Each branch captures features at different receptive field scales.
    Two CPCA blocks provide attention-based feature refinement.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvModule, self).__init__()
        middim = out_channels * 4

        self.bnrelu1 = BNPReLU(nIn=in_channels)

        # 1x1 convolution for channel expansion
        self.conv1x1_1 = nn.Conv2d(in_channels, middim, kernel_size=1, dilation=1)

        # Branch 1: small receptive field (dilation=3)
        self.conv1x3 = nn.Conv2d(middim, middim, kernel_size=(1, 3), dilation=1, padding=(0, 1), groups=middim)
        self.conv3x1 = nn.Conv2d(middim, middim, kernel_size=(3, 1), dilation=1, padding=(1, 0), groups=middim)
        self.conv3x3_1 = nn.Conv2d(middim, middim, kernel_size=3, dilation=3, padding=3, groups=middim)

        # Branch 2: medium receptive field (dilation=5)
        self.conv1x5 = nn.Conv2d(middim, middim, kernel_size=(1, 5), dilation=1, padding=(0, 2), groups=middim)
        self.conv5x1 = nn.Conv2d(middim, middim, kernel_size=(5, 1), dilation=1, padding=(2, 0), groups=middim)
        self.conv3x3_2 = nn.Conv2d(middim, middim, kernel_size=3, dilation=5, padding=5, groups=middim)

        # Branch 3: large receptive field (dilation=7)
        self.conv1x7 = nn.Conv2d(middim, middim, kernel_size=(1, 7), dilation=1, padding=(0, 3), groups=middim)
        self.conv7x1 = nn.Conv2d(middim, middim, kernel_size=(7, 1), dilation=1, padding=(3, 0), groups=middim)
        self.conv3x3_3 = nn.Conv2d(middim, middim, kernel_size=3, dilation=7, padding=7, groups=middim)

        # CPCA attention blocks for feature refinement
        self.eca_1 = CPCABlock(middim, middim)
        self.eca_2 = CPCABlock(middim, middim)

        # Final 3x3 convolution to fuse all branches
        self.conv3x3_final = nn.Conv2d(middim * 3, out_channels, kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x):
        residual = x
        x = self.bnrelu1(x)

        # Channel expansion
        x1 = self.conv1x1_1(x)

        # Attention-based feature refinement
        x_eca_1 = self.eca_1(x1)
        x_eca_2 = self.eca_2(x1)

        # Multi-scale branches
        x2 = self.conv1x3(x1)
        x2 = self.conv3x1(x2)
        x2 = self.conv3x3_1(x2)

        x3 = self.conv1x5(x1)
        x3 = self.conv5x1(x3)
        x3 = self.conv3x3_2(x3)

        x4 = self.conv1x7(x1)
        x4 = self.conv7x1(x4)
        x4 = self.conv3x3_3(x4)

        # Feature fusion with cross-branch attention
        x_branch1 = x2 + x_eca_1 + x3
        x_branch2 = x3 + x_eca_2 + x4

        # Concatenate and reduce
        out = torch.cat([x_branch2, x_branch1, x1], dim=1)
        out = self.conv3x3_final(out)

        return out


if __name__ == '__main__':
    input = torch.randn(1, 128, 224, 224).to('cuda')
    model = MultiScaleConvModule(in_channels=128, out_channels=256).to('cuda')
    out = model(input)
    print("Output shape:", out.shape)
