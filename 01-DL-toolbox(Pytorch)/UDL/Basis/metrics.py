# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import torch
import numpy as np
import math
import torch.nn as nn


def rgb2ycbcr(img, y_only=True):
    """metrics"""
    img.astype(np.float32)
    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt


def quantize(img, rgb_range):
    pixel_range = 255.0 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range):
    """metrics"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    # .reshape((1, 1, 3)) / 256#
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)  # (1)
    if hr.size == 1:
        return 0
    if scale != 1:
        shave = scale
    else:
        shave = scale + 6
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
        # valid = diff[shave:-shave, shave:-shave, ...]
    # mse = np.mean(np.mean(pow(valid, 2), axis=[1, 2, 3]), axis=0)
    mse = np.mean(pow(valid, 2))
    if mse == 0:
        return 100
    try:
        psnr = -10 * math.log10(mse)
    except Exception:
        print(mse)

    return psnr


class PSNR_ycbcr(nn.Module):

    def __init__(self):
        super().__init__()
        self.gray_coeffs = torch.tensor([65.738, 129.057, 25.064],
                                        requires_grad=False).reshape((1, 3, 1, 1)) / 256

    def quantize(self, img, rgb_range):
        """metrics"""
        pixel_range = 255 / rgb_range
        img = torch.multiply(img, pixel_range)
        img = torch.clip(img, 0, 255)
        img = torch.round(img) / pixel_range
        return img

    @torch.no_grad()
    def forward(self, sr, hr, scale, rgb_range):
        """metrics"""
        sr = self.quantize(sr, rgb_range)
        gray_coeffs = self.gray_coeffs.to(sr.device)

        hr = hr.float()
        sr = sr.float()
        diff = (sr - hr) / rgb_range

        diff = torch.multiply(diff, gray_coeffs).sum(1)
        if hr.size == 1:
            return 0
        if scale != 1:
            shave = scale
        else:
            shave = scale + 6
        if scale == 1:
            valid = diff
        else:
            valid = diff[..., shave:-shave, shave:-shave]
        mse = torch.mean(torch.pow(valid, 2))
        return -10 * torch.log10(mse)


def sub_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x / 255.0


def add_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x