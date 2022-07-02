# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
from UDL.Basis.python_sub_class import PanSharpeningModel, TaskDispatcher, ModelDispatcher
import UDL.Basis.option

def build_model(arch, task, cfg=None):

    if task == "pansharpening":
        from UDL.pansharpening.models import PanSharpeningModel as MODELS

        return MODELS.build_model(cfg)
    else:
        raise NotImplementedError(f"It's not supported in {task}")


def getDataSession(cfg):

    task = cfg.task

    if task in ["pansharpening"]:
        from UDL.pansharpening.common.psdata import PansharpeningSession as DataSession
    else:
        raise NotImplementedError

    return DataSession(cfg)
