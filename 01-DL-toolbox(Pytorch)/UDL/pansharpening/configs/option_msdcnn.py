import argparse
# from UDL.Basis.option import panshaprening_cfg, Config, os
from UDL.AutoDL import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='MSDCNN'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()
        if cfg is None:
            from UDL.Basis.option import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0]

        model_path = f'{root_dir}/results/{cfg.task}/wv3/MSDCNN/Test/.pth.tar'

        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        # * Training
        parser.add_argument('--lr', default=0.000001, type=float)  # 1e-4 2e-4 8
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=64, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--epochs', default=500, type=int)
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='MSDCNN', type=str,
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
        args.experimental_desc = "Test"
        # cfg.save_fmt = 'png'
        cfg.img_range = 2047.0

        cfg.merge_args2cfg(args)
        print(cfg.pretty_text)
        # cfg.workflow = [('train', 50), ('val', 1)]
        # cfg.workflow = [('val', 1)]
        cfg.workflow = [('train', 50)]
        self.merge_from_dict(cfg)

