# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import torch
from UDL.pansharpening.evaluation.ps_evaluate import analysis_accu, save_results
from UDL.Basis.config import Config
import warnings
import scipy.io as sio

class TaskDispatcher(Config):
    _task = dict()

    def __init_subclass__(cls, name='', **kwargs):
        super().__init_subclass__(**kwargs)

        if name != '':
            cls._task[name] = cls
            cls._name = name
            # print(cls.__repr__, cls..__repr__)
        else:
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
            cls._task[cls.__name__] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        if cls is TaskDispatcher:
            task = kwargs.get('task')
            try:
                cls = cls._task[task]
            except KeyError:
                raise ValueError(f'Got task={task} but expected'
                                 f'one of {cls._task.keys()}')

        instance = super().__new__(cls)

        return instance

    # def __len__(self):
    #     return len(self._cfg_dict)
    #
    # def __getattr__(self, name):
    #     return getattr(self._cfg_dict, name)
    #
    # def __delattr__(self, name):
    #     return delattr(self._cfg_dict, name)
    #
    # def __getitem__(self, name):
    #     return self._cfg_dict.__getitem__(name)
    #
    # def __iter__(self):
    #     return iter(self._cfg_dict)
    #
    # def __repr__(self):
    #     return f'TaskDispatcher {self._cfg_dict.__repr__()}'

    # def __setattr__(self, name, value):
    #     if isinstance(value, dict):
    #         value = ConfigDict(value)
    #     print("__setattr__")
    #     self._cfg_dict.__setattr__(name, value)

    # def __setitem__(self, name, value):
    #     if isinstance(value, dict):
    #         value = ConfigDict(value)
    #     print("__setitem__")
    #     self._cfg_dict.__setitem__(name, value)

    @classmethod
    def new(cls, **kwargs):
        # 需要从外部启动和从任务启动，但参数不同
        key = 'mode'
        value = kwargs.setdefault('mode', None)
        print('111', value)
        if value is None:
            # 第二、三调用层进入此函数
            key = 'task'
            if kwargs.get('task', None):
                # 二
                value = kwargs.pop('task')
                print('222', value)
            elif kwargs.get('arch', None):
                # 三
                key = 'arch'
                value = kwargs.pop('arch')
                print('333', value)
            else:
                key = 'arch'

        kwargs.pop('mode')

        try:
            cls = cls._task[value]
        except KeyError:
            warning = f'Got {key}={value} but expected ' \
                      f'one of {cls._task.keys()}'
            warnings.warn(warning)
            return Config()

        return cls(**kwargs)

class ModelDispatcher(object):
    _task = dict()

    def __init_subclass__(cls, name='', **kwargs):
        super().__init_subclass__(**kwargs)
        if name != '':
            cls._task[name] = cls
            cls._name = name
            # print(cls.__repr__, cls..__repr__)
        else:
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
            cls._task[cls.__name__] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        if cls is ModelDispatcher:
            task = kwargs.get('task')
            try:
                cls = cls._task[task]
            except KeyError:
                raise ValueError(f'Got task={task} but expected'
                                 f'one of {cls._task.keys()}')

        instance = super().__new__(cls)

        return instance

    @classmethod
    def build_model(cls, cfg):

        arch = cfg.arch
        task = cfg.task
        model_style = cfg.model_style

        try:
            # 获得PansharpeningModel,进行分发
            cls = cls._task[task](None, None)
        except KeyError:
            raise ValueError(f'Got task={task} but expected '
                             f'one of {cls._task.keys()} in {cls}')
        try:
            # 获得具体的模型
            cls_arch = cls._models[arch]()
        except KeyError:
            raise ValueError(f'Got arch={arch} but expected '
                             f'one of {cls._models.keys()} in {cls}')

        model, criterion, optimizer, scheduler = cls_arch(cfg)

        if model_style is None:
            # 获得PansharpeningModel,model+head
            model_style = task

        if model_style is not None:
            try:
                # 获得具体的模型
                model = cls._task[model_style](model, criterion)
            except KeyError:
                raise ValueError(f'Got model_style={model_style} but expected '
                                 f'one of {cls._models.keys()} (merged in _models) in {cls}')

        return model, criterion, optimizer, scheduler

class PanSharpeningModel(ModelDispatcher, name='pansharpening'):

    _models = {}

    def __init__(self, model=None, criterion=None):
        super(PanSharpeningModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.reg = False
        if hasattr(self.model, 'reg'):
            self.reg = self.model.reg

    def __init_subclass__(cls, name='', **kwargs):

        # print(name, cls)
        if name != '':
            cls._models[name] = cls
            cls._name = name
        else:
            cls._models[cls.__name__] = cls
            cls._name = cls.__name__
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')

    def l2_regularization(self, loss_dict, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in self.model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)
        if isinstance(loss_dict, dict):
            loss_dict['loss'] = loss_dict['loss'] + sum(regularizations)
            loss_dict['log_vars'].update(reg_loss=loss_dict['loss'])
        else:
            loss_dict = loss_dict + sum(regularizations)

        return loss_dict

    def train_step(self, *args, **kwargs):

        loss_dict = self.model.train_step(args[0], **kwargs)

        if self.reg:
            return self.l2_regularization(loss_dict)

        return loss_dict

    def val_step(self, *args, **kwargs):
        sr, gt = self.model.val_step(*args, **kwargs)
        result_our = torch.squeeze(sr).permute(1, 2, 0)
        result_our = torch.clip(result_our, 0, 1)
        metrics = analysis_accu(gt.cuda().squeeze(0) / kwargs['img_range'], result_our, 4)
        # result_our = result_our * kwargs['img_range']

        if kwargs['save_fmt'] is not None:
            save_results(kwargs['idx'], kwargs['save_dir'], kwargs['filename'], kwargs['save_fmt'], result_our)

        return {'log_vars': metrics}

