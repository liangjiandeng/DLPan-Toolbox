    # Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler

import torch.nn as nn

from mmcv.runner.dist_utils import master_only
from mmcv.utils.logging import print_log, get_logger, logger_initialized

class BaseModel(nn.Module):
    _task = {}

    def __init_subclass__(cls, name='', **kwargs):
        if name != '':
            # if name in cls._taskhead.keys():
            #     raise ValueError(f'Got name={name} existed'
            #                      f'in{cls._taskhead.keys()}')
            # else:
            cls._task[name] = cls
            cls._name = name
        else:
            # if cls.__name__ in cls._taskhead.keys():
            #     raise ValueError(f'Got cls.__name__={cls.__name__} existed '
            #                      f'in{cls._taskhead.keys()}')
            # else:
            #     warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
            cls._task[cls.__name__] = cls
            cls._name = cls.__name__

    @classmethod
    def build_model(cls, *args, **kwargs):

        # if cls is StreroSRModel:
        model = kwargs.pop('model')
        try:
            cls = cls._models[model]
            # print(cls)
        except KeyError:
            raise ValueError(f'Got model={model} but expected '
                             f'one of {cls._models.keys()}')

        return cls(None, None)


    @classmethod
    def new(cls, *args, **kwargs):
        task = kwargs.pop('task')
        try:
            cls = cls._task[task]
        except KeyError:
            raise ValueError(f'Got task={task} but expected '
                             f'one of {cls._task.keys()}')

        return cls(*args, **kwargs)

class BaseModule(BaseModel, name='BaseModule'):#nn.Module, metaclass=ABCMeta
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """
    # _task = {}
    #
    # def __init_subclass__(cls, name='', **kwargs):
    #     if name != '':
    #         # if name in cls._taskhead.keys():
    #         #     raise ValueError(f'Got name={name} existed'
    #         #                      f'in{cls._taskhead.keys()}')
    #         # else:
    #             cls._task[name] = cls
    #             cls._name = name
    #     else:
    #         # if cls.__name__ in cls._taskhead.keys():
    #         #     raise ValueError(f'Got cls.__name__={cls.__name__} existed '
    #         #                      f'in{cls._taskhead.keys()}')
    #         # else:
    #         #     warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
    #             cls._task[cls.__name__] = cls
    #             cls._name = cls.__name__
    #
    #
    # @classmethod
    # def new(cls, *args, **kwargs):
    #     task = kwargs.pop('task')
    #     try:
    #         cls = cls._task[task]
    #     except KeyError:
    #         raise ValueError(f'Got task={task} but expected '
    #                          f'one of {cls._task.keys()}')
    #
    #     return cls

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'

        from ..cnn import initialize
        from ..cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(
                    'Name of parameter - Initialization information\n')
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f'\n{name} - {param.shape}: '
                        f"\n{self._params_init_info[param]['init_info']} \n")
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f'\n{name} - {param.shape}: '
                    f"\n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


class Sequential(BaseModule, nn.Sequential, name='Sequential'):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList, name='ModuleList'):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class ModuleDict(BaseModule, nn.ModuleDict, name='ModuleDict'):
    """ModuleDict in openmmlab.

    Args:
        modules (dict, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)


class BaseBackbone(BaseModule, name='BaseBackbone'):
    _models = {}

    def __init_subclass__(cls, name='', **kwargs):
        if name != '':
            # if name in cls._models.keys():
            #     raise ValueError(f'Got name={name} existed'
            #                      f'in{cls._models.keys()}')
            # else:
                cls._models[name] = cls
                cls._name = name
        else:
            # if cls.__name__ in cls._models.keys():
            #     raise ValueError(f'Got cls.__name__={cls.__name__} existed'
            #                      f'in{cls._models.keys()}')
            # else:
            #     warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
                cls._models[cls.__name__] = cls
                cls._name = cls.__name__


    @classmethod
    def build_model(cls, *args, **kwargs):
        model = kwargs.pop('model')
        try:
            cls = cls._models[model]
        except KeyError:
            raise ValueError(f'Got models={model} but expected '
                             f'one of {cls._models.keys()}')

        return cls

class BaseLosses(nn.Module):
    _models = {}

    def __init_subclass__(cls, name='', **kwargs):

        # print(name, cls)
        if name != '':
            # if name in cls._models.keys():
            #     raise ValueError(f'Got name={name} existed'
            #                      f'in{cls._models.keys()}')
            # else:
                cls._models[name] = cls
                cls._name = name
        else:
            # if cls.__name__ in cls._models.keys():
            #     raise ValueError(f'Got cls.__name__={cls.__name__} existed'
            #                      f'in{cls._models.keys()}')
            # else:
            #     warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
                cls._models[cls.__name__] = cls
                cls._name = cls.__name__

    @classmethod
    def build_model(cls, *args, **kwargs):
        model = kwargs.pop('model')
        try:
            cls = cls._models[model]
        except KeyError:
            raise ValueError(f'Got models={model} but expected '
                             f'one of {cls._models.keys()}')

        return cls

class BaseNecks(nn.Module):
    _models = {}

    def __init_subclass__(cls, name='', **kwargs):
        # print(name, cls)
        if name != '':
            # if name in cls._models.keys():
            #     raise ValueError(f'Got name={name} existed'
            #                      f'in{cls._models.keys()}')
            # else:
                cls._models[name] = cls
                cls._name = name
        else:
            # if cls.__name__ in cls._models.keys():
            #     raise ValueError(f'Got cls.__name__={cls.__name__} existed'
            #                      f'in{cls._models.keys()}')
            # else:
            #     warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
                cls._models[cls.__name__] = cls
                cls._name = cls.__name__

    @classmethod
    def build_model(cls, *args, **kwargs):
        model = kwargs.pop('model')
        try:
            cls = cls._models[model]
        except KeyError:
            raise ValueError(f'Got models={model} but expected '
                             f'one of {cls._models.keys()}')

        return cls

class BaseNecksV2(BaseModule, BaseNecks, name='BaseNecksV2'):
    '''
    父类的_models, __init_subclass__都会被继承
    '''
    ...

