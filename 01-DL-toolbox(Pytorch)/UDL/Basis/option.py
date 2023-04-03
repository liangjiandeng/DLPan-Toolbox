# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import argparse
import platform
# import warnings
import os
from UDL.AutoDL import TaskDispatcher
from UDL.Basis.config import Config
import warnings

def common_cfg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # * Logger
    parser.add_argument('--use-log', default=True
                        , type=bool)
    parser.add_argument('--log-dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb-dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--use-tfb', default=False, type=bool)

    # * DDP
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', default=0, type=int,
                        help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank")
    parser.add_argument('--backend', default='nccl', type=str,  # gloo
                        help='distributed backend')
    parser.add_argument('--dist-url', default='env://',
                        type=str,  # 'tcp://224.66.41.62:23456'
                        help='url used to set up distributed training')
    # * AMP
    parser.add_argument('--amp', default=None, type=bool,
                        help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')

    # * Training
    parser.add_argument('--accumulated-step', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=0, type=float,
                        help='gradient clipping max norm')

    # * extra
    parser.add_argument('--seed', default=10, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--reg', type=bool, default=True,
                        help='loss with l2 reguliarization for nn.Conv2D, '
                             'which is very important for classical panshrapening!!! ')

    parser.add_argument('--crop_batch_size', type=int, default=128,
                        help='input batch size for-'
                             ' training')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--model_style', type=str, default=None,
                        help='model_style is used to recursive/cascade or GAN training')
    parser.add_argument('--mode', type=str, default=None,
                        help='dataset file extension')
    parser.add_argument('--task', type=str, default=None,
                        help='dataset file extension')
    parser.add_argument('--arch', type=str, default='',
                        help='arch')
    parser.add_argument('--data_dir', type=str, default='',
                        help='root path for dataset')
    args = parser.parse_args()
    args.global_rank = 0
    args.once_epoch = False
    args.reset_lr = False
    args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
    args.save_top_k = 5
    args.save_print_freq = 10
    args.start_epoch = 1
    assert args.accumulated_step > 0
    args.load_model_strict = True
    args.resume_mode = 'best'
    args.validate = False
    args.gpu_ids = [0]
    # args.workflow = []

    return Config(args)

class panshaprening_cfg(TaskDispatcher, name='pansharpening'):


    def __init__(self, cfg=None, arch=None):
        super(panshaprening_cfg, self).__init__()

        import UDL.pansharpening.configs
        import UDL.pansharpening.models

        if cfg is None:
            cfg = common_cfg()

        cfg.scale = [1]
        if cfg.data_dir == '':
            if platform.system() == 'Linux':
                cfg.data_dir = '/Datasets/pansharpening/DLPan'
            if platform.system() == "Windows":
                cfg.data_dir = 'D:/Datasets/pansharpening/DLPan'
        else:
            raise ValueError(f"'cfg.data_dir={cfg.data_dir}' is empty. ")
        
        cfg.best_prec1 = 10000
        cfg.best_prec5 = 10000
        cfg.metrics = 'loss'
        cfg.task = "pansharpening"
        cfg.save_fmt = "mat" # fmt is mat or not mat
        cfg.taskhead = "pansharpening"

        # * Importantly
        warning = f"Note: FusionNet, DiCNN, PNN don't have high-pass filter"
        warnings.warn(warning)
        if arch is not None:
            cfg = self.new(cfg=cfg, arch=cfg.arch)
        self.merge_from_dict(cfg)

def nni_cfg(args):
    if args.mode == 'nni':
        import nni
        tuner_params = nni.get_next_parameter()
        print("launcher: nni is running. \n", tuner_params)
        args.merge_from_dict(tuner_params)
    return args

class get_cfg(TaskDispatcher, name='entrypoint'):
    def __init__(self, task=None, arch=None):
        super(get_cfg, self).__init__()
        args = common_cfg()
        # args.mode = 'nni'
        if arch is not None:
            args.arch = arch
        if args.mode == 'nni':
            args = nni_cfg(args)
        # args.__delattr__('workflow')

        if hasattr(args, 'task'):
            cfg = TaskDispatcher.new(cfg=args, task=task, arch=args.arch)
            cfg.merge_from_dict(args)
        elif task in TaskDispatcher._task.keys():
            cfg = TaskDispatcher.new(cfg=args, task=task, arch=args.arch)
            cfg.merge_from_dict(args)
        else:
            raise ValueError(f"nni starter don't have task={task} but expected"
                             f"one of {super()._task.keys()} in TaskDispatcher")
        # cfg.setdefault('workflow', [])
        cfg = data_cfg(cfg)
        print(cfg.pretty_text)

        self.merge_from_dict(cfg)


def data_cfg(cfg):
    if cfg.get('config', None) is not None:

        if not os.path.isfile(cfg.config):
            raise IOError(f"reading {cfg.config} failed")

        cfg.merge_from_dict(cfg.fromfile(cfg.config))
        if cfg.get('data', None) is not None and callable(cfg.data):
            data_func = cfg.pop('data')
            cfg.merge_from_dict(Config(data_func(cfg.data_dir)))

        cfg.workflow = cfg.get('workflow', [])
        if cfg.get('norm_cfg', None) is not None and cfg.launcher == 'none':
            cfg.norm_cfg = 'BN'

    # modify loading COCO from extern
    # if hasattr(cfg, 'data'):
    #     cfg.data.train['ann_file'] = cfg.data.train['ann_file'].replace('data', cfg.data_dir)
    #     cfg.data.train['img_prefix'] = cfg.data.train['img_prefix'].replace('data', cfg.data_dir)
    #     cfg.data.val['ann_file'] = cfg.data.val['ann_file'].replace('data', cfg.data_dir)
    #     cfg.data.val['img_prefix'] = cfg.data.val['img_prefix'].replace('data', cfg.data_dir)
    #     cfg.samples_per_gpu = cfg.data.samples_per_gpu
    #     cfg.workers_per_gpu = cfg.data.workers_per_gpu

    return cfg
