# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from PIL import Image
import cv2
import os
import numpy as np
import io
import pathlib
import torch
import math

irange = range


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def norm_image(image, factor=255.):
    """
    标准化图像
    :param factor:
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    if factor == 255. or factor == 255:
        image *= factor
        return np.uint8(image)
    else:
        return image


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    # grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    # im_max = np.percentile(grayscale_im, 99)
    # im_min = np.min(grayscale_im)
    # grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    # grayscale_im = np.expand_dims(grayscale_im, axis=0)
    # return grayscale_im

    grayscale_im = np.sum(np.abs(im_as_arr), axis=-1)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=-1)
    return grayscale_im


def apply_gradient_images(gradient, file_name, is_save=False):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    if is_save:
        path_to_file = os.path.join('../results', file_name + '.jpg')
        save_image(gradient, path_to_file)
        return None
    else:
        return gradient


# misc_function
import matplotlib.cm as mpl_color_map
import copy
import PIL


def gen_colormap(input_image, feature, gradient, factor=255):
    if feature.size(0) == 1:
        feature = feature.squeeze(0)
    if gradient.size(0) == 1:
        gradient = gradient.squeeze(0)
    gradient = gradient.cpu().data.numpy()  # [C,H,W]
    weight = np.mean(gradient, axis=(1, 2))  # [C]

    feature = feature.cpu().data.numpy()  # [C,H,W]

    cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
    cam = np.sum(cam, axis=0)  # [H,W]
    cam = np.maximum(cam, 0)  # ReLU

    # 数值归一化
    # cam -= np.min(cam)
    # cam /= np.max(cam)

    # cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * factor)  # Scale between 0-255 to visualize
    cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                input_image.shape[3]), Image.ANTIALIAS)) / factor
    return cam


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """

    if not isinstance(org_im, np.ndarray):
        org_im = org_im[0, :3, ...].permute(1, 2, 0)
        org_im = org_im.cpu().numpy() * 255
        org_im = PIL.Image.fromarray(org_im.astype(np.uint8))
    else:
        org_im = org_im * 255
        org_im = PIL.Image.fromarray(org_im.astype(np.uint8))
    # Get colormap
    '''
    
    '''
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[..., 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap = heatmap.resize(org_im.size, Image.ANTIALIAS)
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


# hook_test/Viz/main

def gen_grad_cam(image, feature, gradient):
    if feature.size(0) == 1:
        feature = feature.squeeze(0)
    if gradient.size(0) == 1:
        gradient = gradient.squeeze(0)
    gradient = gradient  # .cpu().data.numpy()  # [C,H,W]
    weight = torch.mean(gradient, dim=(1, 2))  # [C]

    # feature = feature.cpu().data.numpy()  # [C,H,W]

    # cam = feature * weight[:, np.newaxis, np.newaxis]
    cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
    cam = torch.maximum(cam, torch.zeros_like(cam))  # ReLU
    cam = torch.sum(cam, dim=0)  # [H,W]
    # cam = torch.mean(feature, dim=0)
    # cam = torch.maximum(cam, torch.zeros_like(cam))  # ReLU

    # 数值归一化
    cam -= torch.min(cam)
    cam /= (torch.max(cam) - torch.min(cam) + 1e-8)

    return cam.cpu().data.numpy()


def apply_heatmap(image, mask, factor=255):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """

    # mask转为heatmap
    if not isinstance(image, np.ndarray):
        image = image[0, :3, ...].permute(1, 2, 0)
        image = image.cpu().numpy()
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()
    # heatmaps = np.tile(np.zeros_like(mask)[..., np.newaxis], [1, 1, 1, 3])
    # for c_idx in range(mask.shape[0]):
    #     c_mask = mask[c_idx, ..., np.newaxis]
    #     heatmap = cv2.applyColorMap(np.uint8(255 * c_mask), cv2.COLORMAP_JET)
    #     heatmaps[c_idx, ...] = np.float32(heatmap) / 255
    #     heatmaps = heatmaps[..., ::-1]  # gbr to rgb
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb
    # 合并heatmap到原始图像
    cam = cv2.resize(heatmap, image.shape[:2]) + np.float32(image)
    return norm_image(cam, 2048), (heatmap * 255).astype(np.uint8)
    # cam = heatmaps[np.newaxis, ...] + np.float32(image)
    # return norm_image(cam), (heatmaps * 255).astype(np.uint8)


def showimage8(images, unnormlize=2047.0, first_channel=False):
    assert images.shape[1] >= 3, print("input images format is not suitable")

    if isinstance(images, torch.Tensor):
        unnormlize = np.where(max(np.float(torch.max(images)), 1.0) > 1.0, 1.0, unnormlize)
        if first_channel:
            images = images.permute(1, 2, 0)
        output = images[..., [0, 2, 4]] * torch.tensor(unnormlize)
        output = torch.clamp(output, 0, 2047)
        output = output.cpu().detach().numpy()

    norm_image = linstretch(output)
    return norm_image[:, :, ::-1]


def linstretch(images, tol=None):
    '''
    NM = N*M;
    for i=1:3
        b = reshape(double(uint16(ImageToView(:,:,i))),NM,1);
        [hb,levelb] = hist(b,max(b)-min(b));
        chb = cumsum(hb);#沿第一个非单一维运算。matlab矩阵顺序 HxWxC,列的累计和
        t(1)=ceil(levelb(find(chb>NM*tol(i,1), 1 )));
        t(2)=ceil(levelb(find(chb<NM*tol(i,2), 1, 'last' )));
        %t(2) = 1;
        b(b<t(1))=t(1);
        b(b>t(2))=t(2);
        b = (b-t(1))/(t(2)-t(1));
        ImageToView(:,:,i) = reshape(b,N,M);
    end
    '''
    # images = np.random.randn(64, 64, 3) * 2047.0
    if tol is None:
        tol = [0.01, 0.995]
    if images.ndim == 3:
        h, w, channels = images.shape
    else:
        images = np.expand_dims(images, axis=-1)
        h, w, channels = images.shape
    N = h * w
    for c in range(channels):
        image = np.float32(np.round(images[:, :, c])).reshape(N, 1)
        hb, levelb = np.histogram(image, bins=math.ceil(image.max() - image.min()))
        chb = np.cumsum(hb, 0)
        levelb_center = levelb[:-1] + (levelb[1] - levelb[0]) / 2
        lbc_min, lbc_max = levelb_center[chb > N * tol[0]][0], levelb_center[chb < N * tol[1]][-1]
        image = np.clip(image, a_min=lbc_min, a_max=lbc_max)
        image = (image - lbc_min) / (lbc_max - lbc_min)
        images[..., c] = np.reshape(image, (h, w))

    images = np.squeeze(images)

    return images


def make_grid(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        mode: str = "grey",
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
) -> torch.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        mode (str, optional): 人为设定通道模式
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        if mode == "RGB":
            tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def tensor_save_image(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        fp: Union[Text, pathlib.Path, BinaryIO],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        format: Optional[str] = None,
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


if __name__ == "__main__":
    a = np.random.randn(3, 3)
    linstretch(a)
