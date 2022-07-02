import os
import subprocess
import torch
from torch import nn
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from logging import info as log_string
try:
    from apex.parallel.distributed import DistributedDataParallel as DDP
except:
    Warning("No module named 'apex")

def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group.
    """
    # There is no need for reduction in the single-proc case
    gpus = dist.get_world_size()
    if gpus == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / gpus)
    return tensors

def init_dist(launcher, args, backend='nccl', **kwargs):
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    # print(f"DDP: {dist.is_available()} {world_size}")
    return rank, world_size

def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None, **kwargs):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # print(proc_id, ntasks, node_list, addr)
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    # print(os.environ)
    dist.init_process_group(backend=backend)

def reduce_mean(tensor, nprocs=None):
    if nprocs is None:
        _, nprocs = get_dist_info()
        if nprocs == 1:
            return tensor
    # print("reduce_mean", tensor)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # print(rt, nprocs)
    rt /= nprocs
    # print(rt)
    return rt

class MMDistributedDataParallel(DistributedDataParallel):

    def __init__(self, model, device_ids):
        super(MMDistributedDataParallel, self).__init__(model, device_ids, find_unused_parameters=True)

        self.ddp = model

    def reduce_mean(self, tensor, nprocs=None):
        if nprocs is None:
            _, nprocs = get_dist_info()
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def ddp_step(self, loss_dicts):
        losses = {}
        _, world_size = get_dist_info()
        if world_size == 1:
            return loss_dicts
        dist.barrier()
        # keys = loss_dicts.keys()
        # reduced_loss = scaled_all_reduce(loss_dicts.values())
        # losses = {k: v for k, v in zip(keys, reduced_loss)}
        for k, loss in loss_dicts.items():
            reduced_loss = self.reduce_mean(loss)
            losses.update({k: reduced_loss})
        return losses

def dist_train_v1(args, model):
    if args.mode == "DDP":
        if args.global_rank == 0:
            log_string(f'Distributed training: {args.distributed}')
        if args.distributed:
            if args.amp is not None:
                if not args.amp:
                    # delay_allreduce delays all communication to the end of the backward pass.
                    log_string("IN apex DistributedDataParallel mode.")
                    model = DDP(model, delay_allreduce=True)
            else:
                # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
                model = MMDistributedDataParallel(model, device_ids=[args.local_rank])
                # train_sampler = torch.auxiliary.data.distributed.DistributedSampler(train_dataset)
                # val_sampler = torch.auxiliary.data.distributed.DistributedSampler(val_dataset)
    elif args.mode == "DP":
        log_string(f'DataParallel training')
        model = nn.DataParallel(model, device_ids=args.device_ids)

    return model

