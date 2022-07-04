# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, Ran Ran, LiangJian Deng
# @reference:
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int
# import sys
# sys.path.append('/home/office-401-remote/桌面/Machine Learning/RanRan')
# print(sys.path)
import torch
import torch.nn as nn
import math
from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer
from UDL.pansharpening.models import PanSharpeningModel

class MSDCNN(nn.Module):
    def __init__(self, spectral_num, criterion, channel=64):
        super(MSDCNN, self).__init__()

        self.criterion = criterion

        input_channel = spectral_num + 1
        output_channel = spectral_num

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=60, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv2_1 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv3 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_3 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv5 = nn.Conv2d(in_channels=30, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.shallow1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.shallow3 = nn.Conv2d(in_channels=32, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, y):  # x: lms; y: pan

        concat = torch.cat([x, y], 1)  # Bsx9x64x64

        out1 = self.relu(self.conv1(concat))  # Bsx60x64x64
        out21 = self.conv2_1(out1)   # Bsx20x64x64
        out22 = self.conv2_2(out1)   # Bsx20x64x64
        out23 = self.conv2_3(out1)   # Bsx20x64x64
        out2 = torch.cat([out21, out22, out23], 1)  # Bsx60x64x64

        out2 = self.relu(torch.add(out2, out1))  # Bsx60x64x64

        out3 = self.relu(self.conv3(out2))  # Bsx30x64x64
        out41 = self.conv4_1(out3)          # Bsx10x64x64
        out42 = self.conv4_2(out3)          # Bsx10x64x64
        out43 = self.conv4_3(out3)          # Bsx10x64x64
        out4 = torch.cat([out41, out42, out43], 1)  # Bsx30x64x64

        out4 = self.relu(torch.add(out4, out3))  # Bsx30x64x64

        out5 = self.conv5(out4)  # Bsx8x64x64

        shallow1 = self.relu(self.shallow1(concat))   # Bsx64x64x64
        shallow2 = self.relu(self.shallow2(shallow1))  # Bsx32x64x64
        shallow3 = self.shallow3(shallow2) # Bsx8x64x64

        out = torch.add(out5, shallow3)  # Bsx8x64x64
        out = self.relu(out)  # Bsx8x64x64

        return out

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                                data['ms'].cuda(), data['pan'].cuda()
        sr = self(lms, pan)

        loss = self.criterion(sr, gt, *args, **kwargs)

        # return sr, loss
        log_vars.update(loss=loss['loss'])
        return {'loss': loss['loss'], 'log_vars': log_vars}

    def val_step(self, data, *args, **kwargs):

        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                                data['ms'].cuda(), data['pan'].cuda()
        sr = self(lms, pan)

        return sr, gt




if __name__ == '__main__':
    lms = torch.randn([1, 8, 64, 64])
    pan = torch.randn([1, 1, 64, 64])
    ms = torch.randn([1, 8, 16, 16])
    model = BDPN(8, None)
    x,_ = model(ms, pan)
    print(x.shape)