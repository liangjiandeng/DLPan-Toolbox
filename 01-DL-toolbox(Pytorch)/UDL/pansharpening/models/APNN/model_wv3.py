import torch
import torch.nn as nn
import math
from variance_sacling_initializer import variance_scaling_initializer
class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=True):
        regularizations = []
        for k, v in model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = criterion + sum(regularizations)
        return loss

def weights_init(m):                                               # 1
    classname = m.__class__.__name__                               # 2
    if classname.find('Conv') != -1:                               # 3
        variance_scaling_initializer(m.weight.data)

# netG.apply(weights_init)                                           # 8


class APNN(nn.Module):
    def __init__(self):
        super(APNN, self).__init__()

        channel = 48
        spectral_num = 8
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize

        '''
        C.Using deeper network
        Finally, during training, we stabilize the layers’
        inputs by means of batch normalization
        
        '''

        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=9, stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        # init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):  # x= lms; y = pan

        #input1 = torch.cat((x, y), 1)  # Bsx9x64x64

        # input1 = self.bn(input1)
        rs = self.relu(self.conv1(x))
        rs = self.relu(self.conv2(rs))
        

        output = self.conv3(rs)

        return output























