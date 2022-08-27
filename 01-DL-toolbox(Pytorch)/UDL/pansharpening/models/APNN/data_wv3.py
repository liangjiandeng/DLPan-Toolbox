import torch.utils.data as data
import torch
import h5py
import numpy as np


class Dataset_Ft(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Ft, self).__init__()


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        # tensor type:
        gt1 = data["gt"][...]
        gt1 = np.array(gt1, dtype=np.float32) / 2047
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        print(self.gt.size())

        lms1 = data["lms"][...]
        lms1 = np.array(lms1, dtype=np.float32) / 2047
        self.lms = torch.from_numpy(lms1)

        ms1 = data["ms"][...]  # NxCxHxW
        ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047  # NxHxWxC
        self.ms = torch.from_numpy(ms1).permute(0, 3, 1, 2) # NxCxHxW:

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047  # NxHxWx1
        pan1 = np.squeeze(pan1, axis=3)  # NxHxW
        pan_tmp = np.expand_dims(pan1, axis=3)   # NxHxWx1
        self.pan = torch.from_numpy(pan_tmp).permute(0, 3, 1, 2) # Nx1xHxW:

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
