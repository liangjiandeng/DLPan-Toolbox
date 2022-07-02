# # Copyright (c) OpenMMLab. All rights reserved.
# import logging
#
# import torch.distributed as dist
#
# logger_initialized = {}
#
#
# def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
#     """Initialize and get a logger by name.
#
#     If the logger has not been initialized, this method will initialize the
#     logger by adding one or two handlers, otherwise the initialized logger will
#     be directly returned. During initialization, a StreamHandler will always be
#     added. If `log_file` is specified and the process rank is 0, a FileHandler
#     will also be added.
#
#     Args:
#         name (str): Logger name.
#         log_file (str | None): The log filename. If specified, a FileHandler
#             will be added to the logger.
#         log_level (int): The logger level. Note that only the process of
#             rank 0 is affected, and other processes will set the level to
#             "Error" thus be silent most of the time.
#         file_mode (str): The file mode used in opening log file.
#             Defaults to 'w'.
#
#     Returns:
#         logging.Logger: The expected logger.
#     """
#     logger = logging.getLogger(name)
#     if name in logger_initialized:
#         return logger
#     # handle hierarchical names
#     # e.g., logger "a" is initialized, then logger "a.b" will skip the
#     # initialization since it is a child of "a".
#     for logger_name in logger_initialized:
#         if name.startswith(logger_name):
#             return logger
#
#     # handle duplicate logs to the console
#     # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
#     # to the root logger. As logger.propagate is True by default, this root
#     # level handler causes logging messages from rank>0 processes to
#     # unexpectedly show up on the console, creating much unwanted clutter.
#     # To fix this issue, we set the root logger's StreamHandler, if any, to log
#     # at the ERROR level.
#     for handler in logger.root.handlers:
#         if type(handler) is logging.StreamHandler:
#             handler.setLevel(logging.ERROR)
#
#     stream_handler = logging.StreamHandler()
#     handlers = [stream_handler]
#
#     if dist.is_available() and dist.is_initialized():
#         rank = dist.get_rank()
#     else:
#         rank = 0
#
#     # only rank 0 will add a FileHandler
#     if rank == 0 and log_file is not None:
#         # Here, the default behaviour of the official logger is 'a'. Thus, we
#         # provide an interface to change the file mode to the default
#         # behaviour.
#         file_handler = logging.FileHandler(log_file, file_mode)
#         handlers.append(file_handler)
#
#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     for handler in handlers:
#         handler.setFormatter(formatter)
#         handler.setLevel(log_level)
#         logger.addHandler(handler)
#
#     if rank == 0:
#         logger.setLevel(log_level)
#     else:
#         logger.setLevel(logging.ERROR)
#
#     logger_initialized[name] = True
#
#     return logger
#
#
# def print_log(msg, logger=None, level=logging.INFO):
#     """Print a log message.
#
#     Args:
#         msg (str): The message to be logged.
#         logger (logging.Logger | str | None): The logger to be used.
#             Some special loggers are:
#             - "silent": no message will be printed.
#             - other str: the logger obtained with `get_root_logger(logger)`.
#             - None: The `print()` method will be used to print log messages.
#         level (int): Logging level. Only available when `logger` is a Logger
#             object or "root".
#     """
#     if logger is None:
#         print(msg)
#     elif isinstance(logger, logging.Logger):
#         logger.log(level, msg)
#     elif logger == 'silent':
#         pass
#     elif isinstance(logger, str):
#         _logger = get_logger(logger)
#         _logger.log(level, msg)
#     else:
#         raise TypeError(
#             'logger should be either a logging.Logger object, str, '
#             f'"silent" or None, but got {type(logger)}')


# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2022/1/24 11:03
# @Author  : Xiao Wu
# @reference:
# Copyright (c) OpenMMLab. All rights reserved.
import json
from collections import defaultdict
import logging
import os
import functools
import torch.distributed as dist
import colorlog
import time
from pathlib import Path

logger_initialized = {}

log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


# def get_root_logger(name, log_file=None, log_level=logging.INFO):
#     return get_logger('mmcls', log_file, log_level)
def get_root_logger(name=None, cfg=None, cfg_name=None, log_level=logging.INFO):
    return get_logger(name, cfg, cfg_name, log_level)
# TODO: Depre
# the same as "get_root_logger"
def create_logger(cfg=None, cfg_name=None, dist_print=0, log_level=logging.INFO):
    return get_logger(None, cfg, cfg_name, log_level)

@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name, final_log_file, color=True):
    # LOG_DIR = cfg.log_dir
    # LOG_FOUT = open(final_log_file, 'w')
    # head = '%(asctime)-15s %(message)s'

    logging.basicConfig(filename=str(final_log_file).replace('\\', '/'), format='%(message)s', level=logging.INFO)
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # console = logging.StreamHandler()
    # logging.getLogger('').addHandler(console)

    logger = logging.getLogger(name)
    # if name in logger_initialized:
    #     return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    # stream_handler = logging.StreamHandler()
    console = colorlog.StreamHandler()
    handlers = [console]

    # logger.setLevel(logging.INFO)
    # formatter = colorlog.ColoredFormatter(
    #     '%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
    #     log_colors=log_colors_config)  # 日志输出格式

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        # console = colorlog.StreamHandler()
        # console.setLevel(logging.DEBUG)
        handlers.append(console)
        # if color:
        #     formatter = _ColorfulFormatter(
        #         colored("%(message)s", "green")
        #     )
        # else:
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s- %(message)s',
        log_colors=log_colors_config)  # 日志输出格式

    # console.setFormatter(formatter)
    # logger.addHandler(console)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)  # log_level
        logger.addHandler(handler)

    # if rank == 0:
    #     logger.setLevel(logging.INFO)  # log_level
    # else:
    #     logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def get_logger(name=None, cfg=None, cfg_name=None, phase='train', log_level=logging.INFO, file_mode='w'):  # log_file=None,
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    if name in logger_initialized:
        if cfg is None: # cfg.use_log
            return logging.getLogger(name)
        else:
            return None
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            if cfg.use_log:
                return logging.getLogger(name)
            else:
                return None

    logger = None
    tensorboard_log_dir = None
    root_output_dir = Path(cfg.out_dir)
    # set up logger in root_path
    if not root_output_dir.exists():
        # if not dist_print: #rank 0-N, 0 is False
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.dataset
    assert isinstance(dataset, dict), print(f"{dataset}'s type is {type(dataset)}, not a dict. ")
    dataset = dataset.get('train') if dataset.get('train', None) is not None else dataset.get('val')
    model = cfg.arch
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')

    # store all output except tb_log file
    final_output_dir = root_output_dir / dataset / model / cfg_name
    if cfg.eval:
        model_save_tmp = os.path.dirname(cfg.resume_from).split('/')[-1]
    else:
        model_save_tmp = "model_{}".format(time_str)

    model_save_dir = final_output_dir / model_save_tmp
    # if not dist_print:
    print_log('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)


    if cfg.use_log:
        cfg_name = '{}_{}'.format(cfg_name, time_str)
        # a logger to save results
        log_file = '{}_{}.log'.format(cfg_name, phase)
        if cfg.eval:
            final_log_file = model_save_dir / log_file
        else:
            final_log_file = final_output_dir / log_file
            # tensorboard_log
            tensorboard_log_dir = root_output_dir / Path(cfg.log_dir) / dataset / model / cfg_name
            # if not dist_print:
            print_log('=> creating tfb logs {}'.format(tensorboard_log_dir))
            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(name, final_log_file)

    return logger, str(final_output_dir), str(model_save_dir), str(
        tensorboard_log_dir)  # logger,

def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict[int, dict[str, list]]:
            Key is the epoch, value is a sub dict. The keys in each sub dict
            are different metrics, e.g. memory, bbox_mAP, and the value is a
            list of corresponding values in all iterations in this epoch.

            .. code-block:: python

                # An example output
                {
                    1: {'iter': [100, 200, 300], 'loss': [6.94, 6.73, 6.53]},
                    2: {'iter': [100, 200, 300], 'loss': [6.33, 6.20, 6.07]},
                    ...
                }
    """
    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict
