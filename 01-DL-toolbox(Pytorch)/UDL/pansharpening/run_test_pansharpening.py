# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:

import sys
sys.path.append('../..')
from UDL.AutoDL import TaskDispatcher
from UDL.AutoDL.trainer import main

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='MSDCNN')
    # cfg.resume_from = "../pretrained-model/WV3/pannet.pth"
    cfg.eval = True
    cfg.workflow = [('val', 1)]
    print(TaskDispatcher._task.keys())
    main(cfg)
