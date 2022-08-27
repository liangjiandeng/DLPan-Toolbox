import os
import datetime
import torch
import psutil
from collections import defaultdict, deque
import time
import sys
sys.path.append('../..')
sys.path.append('../mmcv')
from mmcv.utils.logging import print_log
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from functools import partial

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# def set_random_seed(seed):
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     cudnn.deterministic = True


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))


# class OrderedAverageMeter(object):
#     def __init__(self):


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None, fmt=":f"):
        # self.name = name
        # self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # def __str__(self):
    #     fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #     return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# class logger():
#     def __init__(self, obj, LOG_DIR, parser):
#         logname = 'log_train' + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')+'.txt'
#         self.LOG_FOUT = open(os.path.join(LOG_DIR, logname), 'w')
#         self.LOG_FOUT.write(str(parser)+'\n')
#     def __call__(self, out_str):
#          self.LOG_FOUT.write(out_str+'\n')
#          self.LOG_FOUT.flush()
#          print(out_str)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None, eval=False):
        if fmt is None:
            if not eval:
                fmt = "{value:.7f} (avg:{avg:.7f})"
            else:
                fmt = "{value:.7f} (avg:{avg:.7f}, std:{std:.7f})"
        self.reset(window_size)
        self.fmt = fmt

    def reset(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.val = value
        self.count += n
        self.total += value * n
        self.avg = self.total / self.count

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.val, self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.val = t[0]
        self.count = int(t[1])
        self.total = t[2]
        self.avg = self.total / self.count

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def std(self):
        return torch.tensor(list(self.deque)).std().item()

    # @property
    # def avg(self):
    #     d = torch.tensor(list(self.deque), dtype=torch.float32)
    #     return d.mean().item()

    # @property
    # def global_avg(self):
    #     return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    #
    # @property
    # def value(self):
    #     return self.deque[-1]

    def __str__(self):
        # return self.fmt.format(
        #     median=self.median,
        #     avg=self.avg,
        #     global_avg=self.global_avg,
        #     max=self.max,
        #     value=self.value)
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            max=self.max,
            value=self.val,
            std=self.std)


class MetricLogger(object):



    def __init__(self, logger=None, delimiter="\t", dist_print=0, window_size=20, eval=False):
        self.meters = defaultdict(partial(SmoothedValue, window_size=window_size, eval=eval))
        self.delimiter = delimiter
        self.dist_print = dist_print
        # self.log = get_root_logger("UDL")
        # self.logger = logger

    # {k:v}打印，对每个k都有val、avg、max、deque属性
    def update(self, **kwargs):
        # dist.barrier()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = torch.mean(v)
                if hasattr(v, 'item'):
                    v = v.item()
            assert isinstance(v, (float, int, str)), print("type: ", type(v))
            self.meters[k].update(v)

    # {k:v}打印，对每个k都有val、avg、max、deque属性
    def update_dict(self, kwargs: dict):
        # dist.barrier()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = torch.mean(v)
                if hasattr(v, 'item'):
                    v = v.item()
            assert isinstance(v, (float, int, str)), print("type: ", type(v))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def clear(self):
        self.meters.clear()

    def log_every(self, iterable, print_freq, header=None):
        i = 1
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}MB'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        # log_string = self.logger.info
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj, i
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable):
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if self.dist_print == 0:
                        print_log(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))

                else:
                    print_log(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.dist_print == 0:
            print_log('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))
