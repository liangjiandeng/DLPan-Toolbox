import torch
import torch.nn as nn
import math

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

# def weights_init(m):                                               # 1
#     classname = m.__class__.__name__                               # 2
#     if classname.find('Conv') != -1:                               # 3
#         variance_scaling_initializer(m.weight.data)

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                variance_scaling_initializer(m.weight)  # method 1: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class APNN(nn.Module):
    def __init__(self):
        super(APNN, self).__init__()

        channel = 48
        spectral_num = 4
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

        init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):  # x= lms; y = pan

        #input1 = torch.cat((x, y), 1)  # Bsx9x64x64

        # input1 = self.bn(input1)
        rs = self.relu(self.conv1(x))
        rs = self.relu(self.conv2(rs))
        

        output = self.conv3(rs)

        return output


# ----------------- End-Main-Part ------------------------------------
# QB
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, 0.001)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor


