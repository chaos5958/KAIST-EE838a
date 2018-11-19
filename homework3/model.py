import torch
import torch.nn as nn
import numpy as np
import pdb

#ResBlock
class ResBlock(nn.Module):
    def __init__(self, num_features_in, num_features_out, bias=True):
        super(ResBlock, self).__init__()
        self.conv_01 = nn.Conv2d(in_channels=num_features_in, out_channels=num_features_out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_02 = nn.Conv2d(in_channels=num_features_out, out_channels=num_features_out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x_ = x
        x = self.conv_01(x)
        x = self.relu(x)
        x = self.conv_02(x)

        return x + x_

#Single-scale network: a building block for a multi-scale network
class SingleNet(nn.Module):
    def __init__(self, scale):
        super(SingleNet, self).__init__()
        assert scale in [1, 2, 4]

        #Head: Conv + ReLU
        if scale == 4:
            conv_head = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        else:
            conv_head = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        relu_head = nn.ReLU(True)
        self.head =  nn.Sequential(*[conv_head, relu_head])

        #Body: Residual blocks
        resblocks = []
        for _ in range(9):
            resblocks.append(ResBlock(64, 64))
        self.body = nn.Sequential(*resblocks)

        #Tail: Conv
        self.tail = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x

#Multi-scale network
class MultiNet(nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()

        #Single networks
        self.network_x4 = SingleNet(4)
        self.network_x2 = SingleNet(2)
        self.network_x1 = SingleNet(1)

        #Upconv
        self.upconv_01 = nn.ConvTranspose2d(3, 3, 4, 2, 1)
        self.upconv_02 = nn.ConvTranspose2d(3, 3, 4, 2, 1)

    def forward(self, x):
        x4_input = x[0]
        x2_input = x[1]
        x1_input = x[2]

        x4_output = self.network_x4(x4_input)
        x4_output_ = self.upconv_01(x4_output)

        x2_input = torch.cat((x2_input, x4_output_), dim=1)
        x2_output = self.network_x2(x2_input)
        x2_output_ = self.upconv_02(x2_output)

        x1_input = torch.cat((x1_input, x2_output_), dim=1)
        x1_output = self.network_x2(x1_input)

        return x4_output, x2_output, x1_output

if __name__ == "__main__":
    device = torch.device("cuda:0")

    model = MultiNet().to(device)
    tensor_01 = torch.FloatTensor(1, 3, 256, 256).to(device)
    tensor_02 = torch.FloatTensor(1, 3, 128, 128).to(device)
    tensor_03 = torch.FloatTensor(1, 3, 64, 64).to(device)

    output = model((tensor_03, tensor_02, tensor_01))
    print(output[0].size())
    print(output[1].size())
    print(output[2].size())
