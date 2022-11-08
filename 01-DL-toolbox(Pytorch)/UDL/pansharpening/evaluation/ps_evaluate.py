import os
import datetime
import imageio
import numpy as np
import cv2
import h5py
import torch
import torch.nn.functional as F
from scipy import io as sio
from torch.utils.data import DataLoader, Dataset
from UDL.Basis.auxiliary import MetricLogger, SmoothedValue, set_random_seed
from UDL.Basis.dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
from UDL.pansharpening.common.evaluate import analysis_accu
from UDL.Basis.postprocess import showimage8
import matplotlib.pyplot as plt
# from UDL.Basis.zoom_image_region import show_region_images
from logging import info as log_string

# dmd
def load_gt_compared(file_path_gt, file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    try:
        gt = torch.from_numpy(data1['gt'] / 2047.0)
    except KeyError:
        print(data1.keys())
    compared_data = torch.from_numpy(data2['output_dmdnet_newdata6'] * 2047.0)
    return gt, compared_data


def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if rs.ndim == 4:
        for b in range(data.shape[0]):
            for i in range(data.shape[1]):
                rs[b, i, :, :] = data[b, i, :, :] - cv2.boxFilter(data[b, i, :, :], -1, (5, 5))
    elif len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))

    return rs


def load_dataset_singlemat_hp(file_path, scale):
    data = sio.loadmat(file_path)  # HxWxC

    # tensor type:
    lms = torch.from_numpy(data['I_MS'] / scale).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms_hp = torch.from_numpy(get_edge(data['I_MS_LR'] / scale)).permute(2, 0, 1).unsqueeze(dim=0)  # CxHxW= 8x64x64
    pan_hp = torch.from_numpy(get_edge(data['I_PAN'] / scale))   # HxW = 256x256
    gt = torch.from_numpy(data['I_GT'] / scale)

    return lms.squeeze().float(), ms_hp.squeeze().float(), pan_hp.float(), gt.float()


def load_dataset_singlemat(file_path, scale):
    data = sio.loadmat(file_path)  # HxWxC
    print("load_dataset_singlemat: ", data.keys())
    # tensor type:
    lms = torch.from_numpy(data['I_MS'] / scale).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['I_MS_LR'] / scale).permute(2, 0, 1).unsqueeze(dim=0)  # CxHxW= 8x64x64

    pan = torch.from_numpy(data['I_PAN'] / scale)  # HxW = 256x256
    if data.get('I_GT', None) is None:
        gt = torch.from_numpy(data['I_MS'] / scale)
    else:
        gt = torch.from_numpy(data['I_GT'] / scale)

    return lms.squeeze().float(), ms.squeeze().float(), pan.float(), gt.float()


def load_dataset_H5_hp(file_path, scale, use_cuda=True):
    data = h5py.File(file_path)  # NxHxWxC
    shape_list = []
    # for k in data.keys():
    #     shape_list.append((k, data[k].shape))
    # print(shape_list)

    # tensor type: NxCxHxW:

    lms = torch.from_numpy(data['lms'][...] / scale).float()#.permute(0, 3, 1, 2)
    ms_hp = torch.from_numpy(get_edge(data['ms'][...] / scale)).float()#.permute(0, 3, 1, 2)  # NxCxHxW:
    mms_hp = torch.nn.functional.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
                                          mode="bilinear", align_corners=True)
    pan = np.squeeze(data['pan'][...])
    pan = pan[:, np.newaxis, :, :]  # NxCxHxW (C=1)
    pan_hp = torch.from_numpy(get_edge(pan / scale)).float()#.permute(0, 3, 1, 2)  # Nx1xHxW:
    if data.get('gt', None) is None:
        gt = torch.from_numpy(data['lms'][...]).float()
    else:
        gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'mms:': mms_hp,
            'ms': ms_hp,
            'pan': pan_hp,
            'gt': gt.permute([0, 2, 3, 1])
            }

def load_dataset_H5(file_path, scale, use_cuda=True):
    data = h5py.File(file_path)  # CxHxW
    print(data.keys())
    # tensor type:
    if use_cuda:
        lms = torch.from_numpy(data['lms'][...] / scale).cuda().float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).cuda().float()  # CxHxW= 8x64x64
        pan = torch.from_numpy(data['pan'][...] / scale).cuda().float()  # HxW = 256x256

        gt = torch.from_numpy(data['gt'][...]).cuda().float()

    else:
        lms = torch.from_numpy(data['lms'][...] / scale).float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).float()  # CxHxW= 8x64x64
        pan = torch.from_numpy(data['pan'][...] / scale).float()  # HxW = 256x256
        if data.get('gt', None) is None:
            gt = torch.from_numpy(data['lms'][...]).float()
        else:
            gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'ms': ms,
            'pan': pan,
            'gt': gt.permute([0, 2, 3, 1])
            }


class MultiExmTest_h5(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, suffix='.h5'):
        super(MultiExmTest_h5, self).__init__()

        # self.scale = 2047.0
        # if 'gf' in dataset_name:
        #     self.scale = 1023.0
        self.img_scale = img_scale
        print(f"loading MultiExmTest_h5: {file_path} with {img_scale}")
        # 一次性载入到内存
        if 'hp' not in dataset_name:
            data = load_dataset_H5(file_path, img_scale, False)

        elif 'hp' in dataset_name:
            file_path = file_path.replace('_hp', '')
            data = load_dataset_H5_hp(file_path, img_scale, False)

        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError
        if suffix == '.mat':
            self.lms = data['lms'].permute(0, 3, 1, 2)  # CxHxW = 8x256x256
            self.ms = data['ms'].permute(0, 3, 1, 2)  # CxHxW= 8x64x64
            self.pan = data['pan'].unsqueeze(1)
            self.gt = data['gt'].permute(0, 3, 1, 2)
        else:
            self.lms = data['lms']
            self.ms = data['ms']
            self.pan = data['pan']
            self.gt = data['gt']

        print(f"lms: {self.lms.shape}, ms: {self.ms.shape}, pan: {self.pan.shape}, gt: {self.gt.shape}")

    def __getitem__(self, item):
        return {'lms': self.lms[item, ...],
                'ms': self.ms[item, ...],
                'pan': self.pan[item, ...],
                'gt': self.gt[item, ...]
                }

    def __len__(self):
        return self.gt.shape[0]


class SingleDataset(Dataset):



    def __init__(self, file_lists, dataset_name, img_scale, dataset=None):

        self.img_scale = img_scale
        self.file_lists = file_lists
        print(f"loading SingleDataset: {file_lists} with {img_scale}")
        self.file_nums = len(file_lists)
        self.dataset = {}
        self.dataset_name = dataset_name

        if 'hp' not in dataset_name:
            self.dataset = load_dataset_singlemat
        elif 'hp' in dataset_name:
            self.dataset = load_dataset_singlemat_hp
        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError

    def __getitem__(self, idx):
        file_path = self.file_lists[idx % self.file_nums]
        test_lms, test_ms, test_pan, gt = self.dataset(file_path, self.img_scale)

        if 'hp' not in self.dataset_name:
            return {'gt': gt,
                    'lms': test_lms,
                    'ms': test_ms,
                    'pan': test_pan.unsqueeze(dim=0),
                    'filename': file_path}
        else:
            return {'gt': gt,
                    'lms': test_lms,
                    'ms': test_ms,
                    'pan': test_pan.unsqueeze(dim=0),
                    'filename': file_path}

    def __len__(self):
        return self.file_nums


def save_results(idx, save_model_output, filename, save_fmt, output):
    if filename is None:
        save_name = os.path.join(f"{save_model_output}",
                                 "output_mulExm_{}.mat".format(idx))
        sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})
    else:
        filename = os.path.basename(filename).split('.')[0]
        if save_fmt != 'mat':
            output = showimage8(output)
            filename = '/'.join([save_model_output, filename + ".png"])
            # plt.imsave(filename, output, dpi=300)
            # show_region_images(output, xywh=[50, 100, 50, 50], #sub_width="20%", sub_height="20%",
            #                    sub_ax_anchor=(0, 0, 1, 1))
            # mpl_save_fig(filename)
        else:
            filename = '/'.join([save_model_output, "output_" + filename + ".mat"])
            sio.savemat(filename, {'sr': output.cpu().detach().numpy()})


def mpl_save_fig(filename):
    plt.savefig(f"{filename}", format='svg', dpi=300, pad_inches=0, bbox_inches='tight')


