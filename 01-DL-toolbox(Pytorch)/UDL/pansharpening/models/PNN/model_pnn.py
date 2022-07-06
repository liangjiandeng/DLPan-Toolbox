import torch
import torch.nn as nn
from torch.nn import functional as F
import math
# from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer

class PNN(nn.Module):
    def __init__(self, spectral_num, criterion, channel=64):
        super(PNN, self).__init__()

        self.criterion = criterion

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=9, stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        # init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):  # x = cat(lms,pan)
        input1 = x  # Bsx9x64x64

        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        output = self.conv3(rs)

        return output

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()
        blk = self.blk

        gt = gt[:, :, blk:-blk, blk:-blk]
        lms = torch.cat([lms, pan], dim=1)

        sr = self(lms)

        loss = self.criterion(sr, gt, *args, **kwargs)

        # return sr, loss
        log_vars.update(loss=loss['loss'])
        return {'loss': loss['loss'], 'log_vars': log_vars}

    def val_step(self, data, *args, **kwargs):
        blk = self.blk
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()
        test_I_in1 = torch.cat([lms, pan], dim=1)
        test_I_in1 = F.pad(test_I_in1, (blk, blk, blk, blk), mode='replicate')
        sr = self(test_I_in1)

        return sr, gt

    @classmethod
    def set_blk(cls, blk):
        cls.blk = blk

# ----------------- End-Main-Part ------------------------------------
