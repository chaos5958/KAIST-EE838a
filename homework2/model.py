import torch
import torch.nn as nn
import numpy as np
import pdb

#Encoder: convolution + ReLU layer
class Conv_ReLU_Block(nn.Module):
    def __init__(self, num_features_in, num_features_out, bias=True):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_features_in, out_channels=num_features_out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))

#Encoder: convolution + ReLU layer
class Conv_ReLU_Pool_Block(nn.Module):
    def __init__(self, num_features_in, num_features_out, bias=True):
        super(Conv_ReLU_Pool_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_features_in, out_channels=num_features_out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))

#Decoder: skip-connection layer
class Skip_Conv(nn.Module):
    def __init__(self, num_features_in, num_features_out, bias=True):
        super(Skip_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_features_in * 2, out_channels=num_features_out, kernel_size=1, stride=1, padding=0, bias=bias)

        #Weight initialization: TODO

    def forward(self, x_input, x_skip):
        #Process a skip connection output
        x_input = torch.log(torch.pow(x_skip, 2) + 0.01)
        x = torch.cat((x_input, x_skip), dim=1)
        x = self.conv(x)

        return x

#Decoder: deconvolution layer
class Deconv(nn.Module):
    def __init__(self, num_features_in, num_features_out, bias=True):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=num_features_in, out_channels=num_features_out, kernel_size=4, stride=2, padding=1)
        self.batch = nn.BatchNorm2d(num_features_out)
        self.relu= nn.ReLU(True)

        #Weight initialization: TODO

    def forward(self, x):
        return self.relu(self.batch(self.deconv(x)))

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        #LDR encoder
        encode_1_1 = Conv_ReLU_Block(3, 64)
        encode_1_2 = Conv_ReLU_Block(64, 64)
        encode_2_1 = Conv_ReLU_Pool_Block(64, 128)
        encode_2_2 = Conv_ReLU_Block(128, 128)
        encode_3_1 = Conv_ReLU_Pool_Block(128, 256)
        encode_3_2 = Conv_ReLU_Block(256, 256)
        encode_3_3 = Conv_ReLU_Block(256, 256)
        encode_4_1 = Conv_ReLU_Pool_Block(256, 512)
        encode_4_2 = Conv_ReLU_Block(512, 512)
        encode_4_3 = Conv_ReLU_Block(512, 512)
        encode_5_1 = Conv_ReLU_Pool_Block(512, 512)
        encode_5_2 = Conv_ReLU_Block(512, 512)
        encode_5_3 = Conv_ReLU_Block(512, 512)

        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(*[encode_1_1, encode_1_2]))
        self.encoder.append(nn.Sequential(*[encode_2_1, encode_2_2]))
        self.encoder.append(nn.Sequential(*[encode_3_1, encode_3_2, encode_3_3]))
        self.encoder.append(nn.Sequential(*[encode_4_1, encode_4_2, encode_4_3]))
        self.encoder.append(nn.Sequential(*[encode_5_1, encode_5_2, encode_5_3]))

        #Latent Representation
        self.latent = []
        self.latent.append(Conv_ReLU_Pool_Block(512, 512))
        self.latent.append(Conv_ReLU_Block(512, 512))
        self.latent.append(Deconv(512, 512))
        self.latent = nn.Sequential(*self.latent)

        #HDR decoder
        self.decoder = nn.ModuleList()
        decode_1_1 = Skip_Conv(512, 512)
        decode_1_2 = Deconv(512, 512)
        decode_2_1 = Skip_Conv(512, 512)
        decode_2_2 = Deconv(512, 256)
        decode_3_1 = Skip_Conv(256, 256)
        decode_3_2 = Deconv(256, 128)
        decode_4_1 = Skip_Conv(128, 128)
        decode_4_2 = Deconv(128, 64)
        decode_5_1 = Skip_Conv(64, 64)
        conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
        batch_norm = nn.BatchNorm2d(3)
        relu = nn.ReLU(True)
        decode_5_2 = nn.Sequential(*[conv, batch_norm, relu])
        decode_6_1 = Skip_Conv(3, 3)

        self.decoder = nn.ModuleList()

        self.decoder.append(nn.ModuleList().extend((decode_1_1, decode_1_2)))
        self.decoder.append(nn.ModuleList().extend((decode_2_1, decode_2_2)))
        self.decoder.append(nn.ModuleList().extend((decode_3_1, decode_3_2)))
        self.decoder.append(nn.ModuleList().extend((decode_4_1, decode_4_2)))
        self.decoder.append(nn.ModuleList().extend((decode_5_1, decode_5_2)))
        self.decoder.append(decode_6_1)

    def forward(self, x, alpha):
        #LDR encoder
        input_image = x
        x = self.encoder[0](x)
        output1 = x
        x = self.encoder[1](x)
        output2 = x
        x = self.encoder[2](x)
        output3 = x
        x = self.encoder[3](x)
        output4 = x
        x = self.encoder[4](x)
        output5 = x

        #Latent Represetation
        x = self.latent(x)

        #HDR decoder
        #pdb.set_trace()
        x = self.decoder[0][0](x, output5)
        x = self.decoder[0][1](x)
        x = self.decoder[1][0](x, output4)
        x = self.decoder[1][1](x)
        x = self.decoder[2][0](x, output3)
        x = self.decoder[2][1](x)
        x = self.decoder[3][0](x, output2)
        x = self.decoder[3][1](x)
        x = self.decoder[4][0](x, output1)
        x = self.decoder[4][1](x)
        x = self.decoder[5](x, input_image)

        #Final
        output = torch.mul(1-alpha, torch.pow(input_image, 2)) + torch.mul(alpha, torch.exp(x))

        return output

if __name__ == "__main__":
    device = torch.device("cuda:3")

    model = UNet().to(device)
    tensor = torch.FloatTensor(1, 3, 320, 320).to(device)

    output = model(tensor)
    print(output.size())
