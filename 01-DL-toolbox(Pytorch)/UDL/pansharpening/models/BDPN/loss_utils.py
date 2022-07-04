import torch
import math
import numpy as np
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# X: (N,3,H,W) a batch of RGB images with values ranging from 0 to 255.
# Y: (N,3,H,W)  ssim_val=ssim(X,Y,data_range=255,size_average=False)
# return (N,) ms_ssim_val=ms_ssim(X,Y,data_range=255,size_average=False)
# #(N,)# or set 'size_average=True' to get a scalar value as loss.ssim_loss=ssim(X,Y,data_range=255,size_average=True)
# return a scalar valuems_ssim_loss=ms_ssim(X,Y,data_range=255,size_average=True)
# or reuse windows with SSIM & MS_SSIM. ssim_module=SSIM(win_size=11,win_sigma=1.5,data_range=255,size_average=True,channel=3)
# ms_ssim_module=MS_SSIM(win_size=11,win_sigma=1.5,data_range=255,size_average=True,channel=3)
# ssim_loss=ssim_module(X,Y)ms_ssim_loss=ms_ssim_module(X,Y)



# def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
#     epsilon = 1e-6
#     if is_mean:
#         loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
#     else:
#         loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
#
#     return loss

def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = torch.mean(torch.mean(torch.sqrt(torch.square(torch.sub(tensor1, tensor2))+epsilon), [2, 3, 1]))
    else:
        loss = torch.mean(torch.sum(torch.sqrt(torch.square(torch.sub(tensor1, tensor2))+epsilon), [2, 3, 1]))
    return loss


# def compute_ergas_loss(tensor1, tensor2):
#     epsilon = 1e-8
#     rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tensor1,tensor2)),[1,2])+epsilon)
#     mean = tf.reduce_mean(tensor2, [1, 2])
#     mean = tf.exp(mean)
#     loss = tf.sqrt(tf.reduce_mean(tf.square(tf.divide(rmse,mean)))+epsilon)
#     return loss

def compute_ergas_loss(tensor1, tensor2):
    epsilon = 1e-8
    rmse = torch.sqrt(torch.mean(torch.square(torch.subtract(tensor1, tensor2)), [2, 3])+epsilon)
    mean = torch.mean(tensor2, [2, 3])
    mean = torch.exp(mean)
    loss = torch.sqrt(torch.mean(torch.square(torch.divide(rmse, mean)))+epsilon)
    return loss

# def compute_spetral_shift_loss(tensor1, tensor2):
#     epsilon = 1e-8
#     size = (int(int(tensor1.get_shape()[1])/4), int(int(tensor1.get_shape()[2])/4))
#     tensor_lr1 = tf.image.resize_images(tensor1, size)
#     tensor_lr2 = tf.image.resize_images(tensor2, size)
#     loss = compute_ergas_loss(tensor_lr1, tensor_lr2)
#     #tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tensor_lr1,tensor_lr2)),[1,2])+epsilon))
#     return loss

def compute_spetral_shift_loss(tensor1, tensor2):
    epsilon = 1e-8
    size = (int(int(tensor1.get_shape()[2])/4), int(int(tensor1.get_shape()[3])/4))
    tensor_lr1 = F.interpolate(tensor1, size)
    tensor_lr2 = F.interpolate(tensor2, size)
    loss = compute_ergas_loss(tensor_lr1, tensor_lr2)
    #tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tensor_lr1,tensor_lr2)),[1,2])+epsilon))
    return loss

# def compute_ssim_loss(tensor1, tensor2):
#     ssim = tf.image.ssim_multiscale(tensor1, tensor2, np.float32(2.0))
#     loss = 1 - tf.reduce_mean(ssim)
#     return loss

def compute_ssim_loss(tensor1, tensor2, channel = 8):
    ssim = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=channel)
    loss = 1 - torch.mean(ssim)
    return loss
