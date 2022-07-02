import torch.nn as nn
import torch
import numpy as np
import inspect


def compute_flops(module, inp, out):
    # print(module.__class__)
    # if 'attn' in module.__name__:
    #     print(module.__class__)
    # print(list(filter(lambda m: not m.startswith("__") and not m.endswith("__") and callable(getattr(module, m)), dir(module))))
    if isinstance(module, nn.Conv2d):
        return compute_Conv2d_flops(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_flops(module, inp, out)
    elif isinstance(module, nn.LayerNorm) or 'LayerNorm' in type(module).__name__:
        return compute_LayerNorm_flops(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_flops(module, inp, out)
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_flops(module, inp, out)
    # elif isinstance(module, nn.Upsample):
    #     return compute_Upsample_flops(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_flops(module, inp, out)
    elif 'SwinTEB' in module.__class__.__name__:#
        return compute_WindowAttention_flops(module, inp, out)
    elif 'XCTEB' in module.__class__.__name__:
        return compute_XCA_flops(module, inp, out)
    elif 'MSA' in module.__class__.__name__:
        return compute_MSA_flops(module, inp, out)
    elif 'cGCN' == module.__class__.__name__:
        return compute_cGCN_flops(module, inp, out)
    elif 'sGCN' == module.__class__.__name__:
        return compute_sGCN_flops(module, inp, out)
    else:
        print(f"[Flops]: {module.__class__.__name__} is not supported!")
        return 0
    pass


def compute_cGCN_flops(module, inp, out):
    batch_size, dim, H, W = inp.size()
    dim = dim // 2
    L = H * W

    # N = window_size ** 2
    # num_patches = H * W // N

    # calculate flops for 1 window with token length of N
    flops = 0
    # qkv = self.qkv(x)
    # flops += N * dim * 3 * dim
    # attn = (q @ k.transpose(-2, -1)) b head c (h w) b head (h w) c
    flops += dim * (dim//2) * L
    #  x = (attn @ v)   b head c c  b head c (h w)
    flops += L * dim * (dim//2)

    return batch_size * flops


def compute_sGCN_flops(module, inp, out):

    batch_size, dim, H, W = inp.size()
    dim = dim // 2
    L = H * W

    # calculate flops for 1 window with token length of N
    flops = 0
    # qkv = self.qkv(x)
    # flops += N * dim * 3 * dim
    # attn = (q @ k.transpose(-2, -1)) b head c (h w) b head (h w) c
    flops += dim * dim * L
    #  x = (attn @ v)   b head c c  b head c (h w)
    flops += L * dim * dim

    return batch_size * flops

def compute_Conv2d_flops(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count
    # k * k * c * H * W * o = (乘法 + 加法 + bias) * active_elements_count
    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_BatchNorm2d_flops(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    in_c, in_h, in_w = inp.size()[1:]
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops

def compute_LayerNorm_flops(module, inp, out):
    # assert isinstance(module, nn.LayerNorm)
    if len(inp.size()) == 3:
        inp = inp.unsqueeze(0)
    if len(out.size()) == 3:
        out = out.unsqueeze(0)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    flops = np.prod(inp.shape)

    return flops

def compute_ReLU_flops(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for s in inp.size()[1:]:
        active_elements_count *= s

    return active_elements_count


def compute_Pool2d_flops(module, inp, out):
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
    assert isinstance(module, nn.Linear)
    if len(inp.size()) > 3:
        inp = inp.reshape(inp.size(0), inp.size(1), -1)
    if len(out.size()) > 3:
        out = out.reshape(out.size(0), out.size(1), -1)
    batch_size = inp.size()[0]
    if len(inp.size()) == 3:# and inp.size(0) == 1:
        inp = inp[0, ...]#.squeeze(0)
    if len(out.size()) == 3:# and out.size(0) == 1:
        out = out[0, ...]#.squeeze(0)
    assert len(inp.size()) == 2 and len(out.size()) == 2

    return batch_size * inp.size()[1] * out.size()[1]

def compute_Upsample_flops(module, inp, out):
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    # for s in output_size.

def compute_MSA_flops(module, inp, out):
    # q = inp[0]
    if isinstance(inp, tuple):
        inp = inp[0]
    if module.__class__.__name__ == "MSA":
        N, batch_size, dim = inp.size()
    elif module.__class__.__name__ == "MSA_BNC":
        batch_size, N, dim = inp.size()


    # window_size = module.window_size
    if hasattr(module, 'num_heads'):
        num_heads = module.num_heads
    elif hasattr(module, 'n_heads'):
        num_heads = module.num_heads
    num_patches = 1#H * W // N
    # num_patches = module.num_patches
    # batch_size /= num_patches# B*nH*nW
    # assert batch_size == 1, print(f"{inp.size()} is not compatiable with {num_patches}")

    # print(inp.size(), out.size(), dir(module))

    # calculate flops for 1 window with token length of N
    flops = 0
    # qkv = self.qkv(x)
    # flops += N * dim * 3 * dim
    # attn = (q @ k.transpose(-2, -1))
    flops += num_heads * N * (dim // num_heads) * N
    #  x = (attn @ v)
    flops += num_heads * N * N * (dim // num_heads)
    # x = self.proj(x)
    # flops += N * dim * dim
    return batch_size * num_patches * flops

def compute_WindowAttention_flops(module, inp, out):
    # inp = inp[0].permute(0, 3, 1, 2) # B, p, L, C
    # out = out.permute(0, 3, 1, 2)

    # dim = out.size(1)
    if isinstance(inp, tuple):
        inp = inp[0]
    # inp = inp[0]
    L = len(inp.size())
    if L == 3:
        batch_size, HW, dim = inp.size()
        H = W = int(np.sqrt(HW))
    elif L == 4:
        batch_size, dim, H, W = inp.size()

    window_size = module.window_size
    num_heads = module.num_heads
    N = window_size ** 2
    num_patches = H * W // N
    # num_patches = module.num_patches
    # batch_size /= num_patches# B*nH*nW
    # assert batch_size == 1, print(f"{inp.size()} is not compatiable with {num_patches}")


    # print(inp.size(), out.size(), dir(module))

    # calculate flops for 1 window with token length of N
    flops = 0
    # qkv = self.qkv(x)
    # flops += N * dim * 3 * dim
    # attn = (q @ k.transpose(-2, -1))
    flops += num_heads * N * (dim // num_heads) * N
    #  x = (attn @ v)
    flops += num_heads * N * N * (dim // num_heads)
    # x = self.proj(x)
    # flops += N * dim * dim
    # module.__base__ = f'{module.__class__.__name__}(dim={dim}, win_size={window_size}, nh={num_heads}, n_p={num_patches}, size=({H}, {W}))'
    # print(f'{module.__class__.__name__}, dim={dim}, win_size={window_size}, num_heads={num_heads},'
    #       f'num_patches={num_patches}, img_size=({H}, {W})')
    return batch_size * num_patches * flops


def compute_XCA_flops(module, inp, out):

    dim = out.size(1)
    batch_size, _, H, W = inp.size()
    if hasattr(module, "window_size"):
        window_size = module.window_size
        N = window_size ** 2
        num_patches = H * W // N
    else:
        num_patches = 1
        window_size = 1
        N = H * W
    # window_size = module.window_size
    num_heads = module.num_heads

    # N = window_size ** 2
    # num_patches = H * W // N

    # calculate flops for 1 window with token length of N
    flops = 0
    # qkv = self.qkv(x)
    # flops += N * dim * 3 * dim
    # attn = (q @ k.transpose(-2, -1)) b head c (h w) b head (h w) c
    flops += num_heads * (dim // num_heads) * (dim // num_heads) * N
    #  x = (attn @ v)   b head c c  b head c (h w)
    flops += num_heads * N * (dim // num_heads) * (dim // num_heads)
    # x = self.proj(x)
    # flops += N * dim * dim
    return batch_size * num_patches * flops