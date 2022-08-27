# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np


def get_edge(data):  # for training: HxWxC
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)

        ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
        ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale  # NxHxWxC
        ms1_tmp = get_edge(ms1)  # NxHxWxC
        self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2)  # NxCxHxW:

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale  # NxHxWx1
        pan1 = np.squeeze(pan1, axis=3)  # NxHxW
        pan_hp_tmp = get_edge(pan1)  # NxHxW
        pan_hp_tmp = np.expand_dims(pan_hp_tmp, axis=3)  # NxHxWx1
        self.pan_hp = torch.from_numpy(pan_hp_tmp).permute(0, 3, 1, 2)  # Nx1xHxW:
        print(
            f"gt: {self.gt.size()}, lms: {self.lms.size()}, pan_hp: {self.pan_hp.size()}, ms_hp: {self.ms_hp.size()} with {img_scale}")

    #####必要函数
    def __getitem__(self, index):
        return {'gt': self.gt[index, :, :, :].float(),
                'lms': self.lms[index, :, :, :].float(),
                'ms_hp': self.ms_hp[index, :, :, :].float(),
                'pan_hp': self.pan_hp[index, :, :, :].float()}

        #####必要函数

    def __len__(self):
        return self.gt.shape[0]
