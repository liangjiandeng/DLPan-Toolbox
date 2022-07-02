import torch.nn.modules as nn
import torch
import cv2
import numpy as np
import h5py
import scipy.io as sio
import os



def load_set(file_path, blk):
    data = sio.loadmat(file_path)  # HxWxC

    # tensor type:
    lms = np.array(data['lms'] / 2047.0, dtype=np.float32)
    pan_hp = np.expand_dims(np.array(data['pan'] / 2047.0,dtype=np.float32), axis=-1)
    lms = np.concatenate([lms, pan_hp], axis=-1)
    lms = np.pad(lms, ((blk, blk), (blk, blk), (0, 0)), mode='edge')
    lms = torch.from_numpy(lms).cuda().permute(2, 0, 1)  # CxHxW = 8x256x256
    pan_hp = torch.from_numpy(pan_hp).cuda().permute(2, 0, 1)   # HxW = 256x256
    ms_hp = torch.from_numpy(data['ms'] / 2047.0).cuda().permute(2, 0, 1)  # CxHxW= 8x64x64
    gt = torch.from_numpy(data['gt'] / 2047.0).cuda()

    return lms, ms_hp, pan_hp, gt

