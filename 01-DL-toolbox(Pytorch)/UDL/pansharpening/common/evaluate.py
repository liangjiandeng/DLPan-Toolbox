# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import math
import torch
import torch.nn.functional as F
import numpy as np


def q2n(gt, x, q_blocks_size, q_shift):
    '''
    '''
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
        x = x.cpu().numpy()

    N, N1, N2, N3 = gt.shape  # 255 255 8
    size2 = q_blocks_size  # 32

    stepx = math.ceil(N1 / q_shift)  # 8
    stepy = math.ceil(N2 / q_shift)  # 8

    if stepy <= 0:
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * q_shift + q_blocks_size - N1  # 1
    est2 = (stepy - 1) * q_shift + q_blocks_size - N2  # 1
    # if np.sum(np.array([est1 != 0, est2 != 0])) > 0:
    # refref = np.zeros(shape=[N1+1, N2+1])
    # fusfus = refref.copy()

    for i in range(N3):
        a1 = gt[..., 0]

        ia1 = np.zeros(shape=[N, N1 + est1, N2 + est2])
        ia1[:, : N1, : N2] = a1
        ia1[:, :, N2:N2 + est2] = ia1[:, :, N2 - 1:-1:N2 - est2 + 1]
        ia1[:, N1:N1 + est1, ...] = ia1[:, N1 - 1:-1:N1 - est1 + 1, ...]
        if i == 0:
            refref = ia1[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
        else:
            refref = np.concatenate([refref, ia1[..., np.newaxis]], axis=-1)
        if i < N3:
            gt = gt[..., 1:]

    gt = refref

    for i in range(N3):

        a2 = x[..., 0]
        ia2 = np.zeros(shape=[N, N1 + est1, N2 + est2])
        ia2[:, : N1, : N2] = a2
        ia2[:, :, N2:N2 + est2] = ia2[:, :, N2 - 1:-1:N2 - est2 + 1]
        ia2[:, N1:N1 + est1, ...] = ia2[:, N1 - 1:-1:N1 - est1 + 1, ...]
        if i == 0:
            fusfus = ia2[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
        else:
            fusfus = np.concatenate([fusfus, ia2[..., np.newaxis]], axis=-1)

        if i < N3:
            x = x[..., 1:]
    x = fusfus

    x = np.array(x, dtype=np.uint16)
    gt = np.array(gt, dtype=np.uint16)

    _, N1, N2, N3 = gt.shape

    if math.ceil(math.log2(N3)) - math.log2(N3) != 0:
        Ndif = pow(2, math.ceil(math.log2(N3))) - N3
        dif = np.zeros(shape=[N, N1, N2, Ndif], dtype=np.uint16)
        gt = np.concatenate(gt, dif, axis=-1)
        x = np.concatenate(x, dif, axis=-1)

    _, _, _, N3 = gt.shape

    valori = np.zeros(shape=[N, stepx, stepy, N3])

    for j in range(stepx):
        for i in range(stepy):
            o = onions_quality(gt[:, j * q_shift:j * q_shift + q_blocks_size,
                               i * q_shift: i * q_shift + size2, :],
                               x[:, j * q_shift:j * q_shift + q_blocks_size,
                               i * q_shift: i * q_shift + size2, :],
                               q_blocks_size)
            valori[:, j, i, :] = o
    q2n_idx_map = np.sqrt(np.sum(valori ** 2, axis=-1))
    # q2n_index = np.mean(q2n_idx_map)
    return q2n_idx_map


def norm_blocco(x, eps=1e-8):
    a = x.mean()
    c = x.std()
    if c == 0:
        c = eps
    return (x - a) / c + 1, a, c


def onions_quality(dat1, dat2, size1):
    dat1 = np.float64(dat1)
    dat2 = np.float64(dat2)

    dat2 = np.concatenate([dat2[..., 0, np.newaxis], -dat2[..., 1:]], axis=-1)
    N, _, _, N3 = dat1.shape
    size2 = size1

    for i in range(N3):
        a1, s, t = norm_blocco(np.squeeze(dat1[..., i]))
        # print(s,t)
        dat1[..., i] = a1
        if s == 0:
            if i == 0:
                dat2[..., i] = dat2[..., i] - s + 1
            else:
                dat2[..., i] = -(-dat2[..., i] - s + 1)
        else:
            if i == 0:
                dat2[..., i] = ((dat2[..., i] - s) / t) + 1
            else:
                dat2[..., i] = -(((-dat2[..., i] - s) / t) + 1)
    m1 = np.zeros(shape=[N, N3])
    m2 = m1.copy()

    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros(shape=[size1, size2])
    mod_q2 = np.zeros(shape=[size1, size2])

    for i in range(N3):
        m1[..., i] = np.mean(np.squeeze(dat1[..., i]))
        m2[..., i] = np.mean(np.squeeze(dat2[..., i]))
        mod_q1m += m1[..., i] ** 2
        mod_q2m += m2[..., i] ** 2
        mod_q1 += np.squeeze(dat1[..., i]) ** 2
        mod_q2 += np.squeeze(dat2[..., i]) ** 2

    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)

    termine2 = mod_q1m * mod_q2m  # 7.97
    termine4 = mod_q1m ** 2 + mod_q2m ** 2  #
    int1 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q1 ** 2)
    int2 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q2 ** 2)
    termine3 = int1 + int2 - (size1 * size2) / ((size1 * size2 - 1)) * (mod_q1m ** 2 + mod_q2m ** 2)  # 17.8988  ** 2
    mean_bias = 2 * termine2 / termine4  # 1
    if termine3 == 0:
        q = np.zeros(shape=[N, 1, N3])
        q[:, :, N3 - 1] = mean_bias
    else:
        cbm = 2 / termine3
        # 32 32 8
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1.reshape(-1), m2.reshape(-1))
        qv = np.zeros(shape=[N, N3])
        for i in range(N3):
            qv[..., i] = (size1 * size2) / ((size1 * size2) - 1) * np.mean(np.squeeze(qu[:, :, i]))
        q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
        q = q * mean_bias * cbm
    return q


def onion_mult2D(onion1, onion2):
    _, _, _, N3 = onion1.shape

    if N3 > 1:
        L = N3 // 2
        a = onion1[..., : L]
        b = onion1[..., L:]
        b = np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1)
        c = onion2[..., : L]
        d = onion2[..., L:]
        d = np.concatenate([d[..., 0, np.newaxis], -d[..., 1:]], axis=-1)

        if N3 == 2:
            ris = np.concatenate([a * c - d * b, a * d + c * b], axis=-1)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1))
            ris3 = onion_mult2D(np.concatenate([a[..., 0, np.newaxis], -a[..., 1:]], axis=-1), d)
            ris4 = onion_mult2D(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis=-1)
    else:
        ris = onion1 * onion2
    return ris


def onion_mult(onion1, onion2):
    # _, N = onion1.shape
    N = len(onion1)
    if N > 1:

        L = N // 2
        a = onion1[:L]
        b = onion1[L:]
        # b[1:] = -b[1:]
        b = np.append(np.array(b[0]), -b[1:])
        c = onion2[:L]
        d = onion2[L:]
        # d[1:] = -d[1:]
        d = np.append(np.array(d[0]), -d[1:])

        if N == 2:
            ris = np.append(a * c - d * b, a * d + c * b)
        else:

            ris1 = onion_mult(a, c)
            # b[1:] = -b[1:]
            ris2 = onion_mult(d, np.append(np.array(b[0]), -b[1:]))
            # a[1:] = -a[1:]
            ris3 = onion_mult(np.append(np.array(a[0]), -a[1:]), d)
            ris4 = onion_mult(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.append(aux1, aux2)
    else:
        ris = np.array(onion1).reshape(-1) * np.array(onion2).reshape(-1)
    return ris


def compute_index(img_base, img_out, ratio):
    h = img_out.shape[0]
    w = img_out.shape[1]
    chanel = img_out.shape[2]
    # 计算SAM
    sum1 = torch.sum(img_base * img_out, 2)
    sum2 = torch.sum(img_base * img_base, 2)
    sum3 = torch.sum(img_out * img_out, 2)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    SAM = averangle * 180 / 3.14159256

    # 计算ERGAS
    summ = 0
    for i in range(chanel):
        a1 = torch.mean((img_base[:, :, i] - img_out[:, :, i]) ** 2)
        m1 = torch.mean(img_base[:, :, i])
        a2 = m1 * m1
        summ = summ + a1 / a2
    ERGAS = 100 * (1 / ratio) * ((summ / chanel) ** 0.5)

    return SAM, ERGAS


import decimal

decimal.getcontext().rounding = "ROUND_HALF_UP"
n_digits = 6


def analysis_accu(img_base, img_out, ratio, flag_cut_bounds=True, dim_cut=21, choices=4):
    if flag_cut_bounds:
        img_base = img_base[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]  #:
        img_out = img_out[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]  #:

    # q2n
    # q2n_index = q2n(img_base, img_out, q_blocks_size=32, q_shift=32)

    h = img_out.shape[0]
    w = img_out.shape[1]
    chanel = img_out.shape[2]

    # 计算SAM
    sum1 = torch.sum(img_base * img_out, 2)
    sum2 = torch.sum(img_base * img_base, 2)
    sum3 = torch.sum(img_out * img_out, 2)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num

    # you can adopt https://segmentfault.com/a/1190000018929994 to compute, too.
    # averangle = math.ceil(averangle * 1000000) / 1000000
    averangle = (averangle * 10 ** n_digits).round() / (10 ** n_digits)
    # SAM = decimal.Decimal(averangle.cpu().numpy() * 180 / 3.14159256).quantize(decimal.Decimal("0.00000"))
    SAM = averangle * 180 / 3.14159256

    # 计算ERGAS
    summ = 0
    for i in range(chanel):
        a1 = torch.mean((img_base[:, :, i] - img_out[:, :, i]) ** 2)
        m1 = torch.mean(img_base[:, :, i])
        a2 = m1 * m1
        summ = summ + a1 / a2
    ERGAS = 100 * (1 / ratio) * ((summ / chanel) ** 0.5)

    # 计算PSNR
    # mse = torch.mean((img_base - img_out) ** 2, 0)
    # mse = torch.mean(mse, 0)
    # rmse = mse ** 0.5
    # temp = torch.log(1 / rmse) / math.log(10)
    PSNR = 10 * torch.log10(math.pow(1.0, 2) / torch.mean((img_out-img_base)**2, [0, 1]))

    # SSIM
    # img_base = img_base.permute(2, 0, 1)
    # img_out = img_out.permute(2, 0, 1)
    # img_base = img_base.unsqueeze(0)
    # img_out = img_out.unsqueeze(0)
    # SSIM = _ssim(img_base.permute(2, 0, 1).unsqueeze(0), img_out.permute(2, 0, 1).unsqueeze(0))

    # index = torch.zeros((5, chanel + 1))
    # index[0, 1:chanel + 1] = CC
    # index[1, 1:chanel + 1] = PSNR
    # index[2, 1:chanel + 1] = SSIM
    # index[0, 0] = torch.mean(CC)
    # index[1, 0] = torch.mean(PSNR)
    # index[2, 0] = torch.mean(SSIM)
    # index[3, 0] = SAM
    # index[4, 0] = ERGAS

    PSNR = torch.mean(PSNR)
    # SSIM = torch.mean(SSIM)
    # q2n_index = np.mean(q2n_index)

    if choices == 5:
        # 计算CC
        C1 = torch.sum(torch.sum(img_base * img_out, 0), 0) - h * w * (
                torch.mean(torch.mean(img_base, 0), 0) * torch.mean(torch.mean(img_out, 0), 0))
        C2 = torch.sum(torch.sum(img_out ** 2, 0), 0) - h * w * (torch.mean(torch.mean(img_out, 0), 0) ** 2)
        C3 = torch.sum(torch.sum(img_base ** 2, 0), 0) - h * w * (torch.mean(torch.mean(img_base, 0), 0) ** 2)
        CC = C1 / ((C2 * C3) ** 0.5)
        CC = torch.mean(CC)
        return {'SAM': SAM, 'ERGAS': ERGAS, 'PSNR': PSNR, 'CC': CC}  # , q2n_index

    return {'SAM': SAM, 'ERGAS': ERGAS, 'PSNR': PSNR, }


def _ssim(img1, img2):
    img1 = img1.float()
    img2 = img2.float()

    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11
    window = create_window(window_size, sigma, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    t = ssim_map.shape
    return ssim_map.mean(2).mean(2)


from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def compare_index(A):
    A_size = A.shape
    ite_n = A_size[2]
    band_n = A_size[1]
    C_better = A[:, 0, 0]
    ind = 0
    for i in range(ite_n):
        score_b = 0
        score_c = 0
        C_compare = A[:, 0, i]
        if (C_better[0] > C_compare[0]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[1] > C_compare[1]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[2] > C_compare[2]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[3] < C_compare[3]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1
        if (C_better[4] < C_compare[4]):
            score_b = score_b + 1
        else:
            score_c = score_c + 1

        if (score_c > score_b):
            C_better = A[:, 0, i]
            ind = i

    C_best = A[:, :, ind]
    best_ind = ind + 1
    return C_best, best_ind


if __name__ == "__main__":
    # a = np.zeros(shape=[256, 256])
    # print(a[:255, :255].shape)
    from scipy import io as sio
    ms = sio.loadmat('../../tests/I_MS.mat')['I_MS'] / 2047.0
    gt = sio.loadmat('../../tests/I_GT.mat')['I_GT'] / 2047.0
    ms = torch.from_numpy(ms).float() #* 2047.0
    gt = torch.from_numpy(gt).float() #* 2047.0
    print(analysis_accu(ms, gt, ratio=4, dim_cut=21))