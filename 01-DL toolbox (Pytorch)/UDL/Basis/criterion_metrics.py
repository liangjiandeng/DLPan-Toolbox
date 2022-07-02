# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
from torch import nn
import torch
from torch import distributed as dist
from collections import OrderedDict

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: n able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relatiumber of object categories, omitting the special no-object category
            matcher: moduleve classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_dicts = {}

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        for k in self.losses.keys():
            # k, loss = loss_dict
            if k == 'loss':
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts#self._parse_losses(self.loss_dicts)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # top-1, top-5 both belong to accuracy
            elif isinstance(loss_value, dict):
                # log_vars[loss_name] = {}
                for name, value in loss_value.items():
                    log_vars[name] = value
                    # log_vars[loss_name].update({name: value.item()})
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        assert 'loss' not in log_vars.keys(), KeyError("key: 'loss' can't be set from cfg_file.")
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'top' not in _key)#if 'loss' in _key
        log_vars['loss'] = loss
        # output = log_vars.pop('acc') #get
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        # log_vars.update(acc=output)

        return loss, log_vars