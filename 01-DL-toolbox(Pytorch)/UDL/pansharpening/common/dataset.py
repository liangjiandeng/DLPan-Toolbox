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

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        if 'valid' in file_path:
            self.gt = self.gt.permute([0, 2, 3, 1])

        print(pan1.shape, lms1.shape, gt1.shape, ms1.shape)
    #####必要函数
    def __getitem__(self, index):
        return {'gt':self.gt[index, :, :, :].float(),
               'lms':self.lms[index, :, :, :].float(),
               'ms':self.ms[index, :, :, :].float(),
               'pan':self.pan[index, :, :, :].float()}

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
