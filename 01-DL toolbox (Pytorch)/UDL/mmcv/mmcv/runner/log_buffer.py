# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import torch
import numpy as np


class LogBuffer:

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    # def update(self, vars, count=1):
    #     assert isinstance(vars, dict)
    #     for key, var in vars.items():
    #         if key not in self.val_history:
    #             self.val_history[key] = []
    #             self.n_history[key] = []
    #         self.val_history[key].append(var)
    #         self.n_history[key].append(count)

    # {k:v}打印，对每个k都有val、avg、max、deque属性
    def update(self, vars, count=1):
        # dist.barrier()
        for k, v in vars.items():
            if k not in self.val_history:
                self.val_history[k] = []
                self.n_history[k] = []
            if isinstance(v, torch.Tensor):
                v = torch.mean(v)
                if hasattr(v, 'item'):
                    v = v.item()
            assert isinstance(v, (float, int, str)), print(f"{k} type: {type(v)}")
            self.val_history[k].append(v)
            self.n_history[k].append(count)

    def average(self, n=0):
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True
