import argparse
# from UDL.Basis.option import panshaprening_cfg, Config, os
from UDL.AutoDL import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='FusionNet'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()
        if cfg is None:
            from UDL.Basis.option import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0]

        model_path = f'./.pth.tar'


        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        parser.add_argument('--mode', default=argparse.SUPPRESS, help='protective declare, please ignore it')

        parser.add_argument('--lr', default=3e-4, type=float)
        # parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=32, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=1, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=400, type=int)
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        ##
        parser.add_argument('--arch', '-a', metavar='ARCH', default='FusionNet', type=str,
                            choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet'])
        parser.add_argument('--dataset', default={'train': 'wv3', 'val': 'NY1_WV3_RR'}, type=str,
                            choices=[None, 'wv2', 'wv3', 'wv4', 'qb',
                                     'TestData_qb', 'TestData_wv2', 'TestData_wv3', 'TestData_wv4',
                                     'San_Francisco_QB_RR', 'San_Francisco_QB_FR', 'NY1_WV3_FR',
                                     'NY1_WV3_RR', 'Alice_WV4_FR', 'Alice_WV4_RR', 'Rio_WV2_FR', 'Rio_WV2_RR'],
                            help="training choices: ['wv2', 'wv3', 'wv4', 'qb'],"
                                 "validation choices: ['valid_wv2_10000','valid_wv3_10000', 'valid_wv4_10000', 'valid_qb_10000']"
                                 "test choices is ['TestData_wv2', 'TestData_wv3', 'TestData_wv4', 'TestData_qb'], and others with RR/FR")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")


        args = parser.parse_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = 'Test'
        cfg.merge_args2cfg(args)
        cfg.save_fmt = "mat"
        # cfg.workflow = [('train', 10), ('val', 1)]
        cfg.workflow = [('val', 1), ('train', 1)]
        # cfg.config = f"{script_path}/configs/hook_configs.py"
        cfg.use_tfb = False
        cfg.img_range = 2047.0#1023.0

        self.merge_from_dict(cfg)