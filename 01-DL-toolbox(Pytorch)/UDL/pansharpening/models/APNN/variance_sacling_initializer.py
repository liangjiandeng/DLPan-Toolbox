import torch
import torch.nn as nn
import math


def truncated_normal_(tensor, mean=0.0, std=1.0):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm
    def calculate_fan(shape, factor=2.0, mode='FAN_IN', uniform=False):
        # 64 9 3 3 -> 3 3 9 64
        # 64 64 3 3 -> 3 3 64 64
        if shape:
            # fan_in = float(shape[1]) if len(shape) > 1 else float(shape[0])
            # fan_out = float(shape[0])
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == 'FAN_IN':
            # Count only number of input connections.
            n = fan_in
        elif mode == 'FAN_OUT':
            # Count only number of output connections.
            n = fan_out
        elif mode == 'FAN_AVG':
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            raise NotImplemented
            # # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            # limit = math.sqrt(3.0 * factor / n)
            # return random_ops.random_uniform(shape, -limit, limit,
            #                                  dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
        return fan_in, fan_out, trunc_stddev

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        # fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        x = x.permute(3, 2, 1, 0)  # .permute(2, 3, 1, 0)
        fan_in, fan_out, trunc_stddev = calculate_fan(x.shape)
        print(trunc_stddev)
        # if mode == "fan_in":
        #     scale /= max(1., fan_in)
        # elif mode == "fan_out":
        #     scale /= max(1., fan_out)
        # else:
        #     scale /= max(1., (fan_in + fan_out) / 2.)
        # if distribution == "normal" or distribution == "truncated_normal":
        #     # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        #     stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, trunc_stddev)  # 0.001)
        x = x.permute(3, 2, 0, 1)
        print(x.min(), x.max())
        return x  # /10*1.28

    variance_scaling(tensor)

    return tensor
