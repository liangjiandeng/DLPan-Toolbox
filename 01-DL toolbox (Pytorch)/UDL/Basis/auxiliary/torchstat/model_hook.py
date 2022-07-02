import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torchstat import compute_madd
from torchstat import compute_flops
from torchstat import compute_memory


class ModelHook(object):
    def __init__(self, model, input_size, device="cuda", debug_layers=[]):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))
        self.leaf_modules = []
        self.debug_layers = debug_layers
        self._model = model
        self._input_size = input_size
        self._origin_call = dict()  # sub module call hook
        self.hooks = []
        self._hook_model()
        # x = [torch.rand(1, *self._input_size)]  # add module duration time
        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        x = [torch.rand(*in_size).type(dtype) for in_size in input_size]
        self._model.eval()
        self._model(*x)

        # if len(debug_layers) > 0:
        #     self.debug_partial_layer(debug_layers)


    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            return

        module.register_buffer('input_shape', torch.zeros(3).int())
        module.register_buffer('output_shape', torch.zeros(3).int())
        module.register_buffer('parameter_quantity', torch.zeros(1).int())
        module.register_buffer('inference_memory', torch.zeros(1).long())
        module.register_buffer('MAdd', torch.zeros(1).long())
        module.register_buffer('duration', torch.zeros(1).float())
        module.register_buffer('Flops', torch.zeros(1).long())
        module.register_buffer('Memory', torch.zeros(2).long())

    def _sub_module_call_hook(self):
        def wrap_call(module, *input, **kwargs):
            assert module.__class__ in self._origin_call
            # Itemsize for memory
            try:
                itemsize = input[0].detach().numpy().itemsize
            except:
                itemsize = input[0].detach().cpu().numpy().itemsize

            start = time.time()
            output = self._origin_call[module.__class__](module, *input, **kwargs)  # 都是nn.Conv2D则有相同的_call__不需要重复存储
            end = time.time()
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))
            # c, h, w
            module.input_shape = torch.from_numpy(
                np.array(input[0].size()[1:], dtype=np.int32))
            module.output_shape = torch.from_numpy(
                np.array(output.size()[1:], dtype=np.int32))
            # print(module.name)
            parameter_quantity = 0
            inference_memory = 1
            # iterate through parameters and count num params
            if 'XCTEB' in module.__class__.__name__:
                c, h, w = module.input_shape
                num_heads = module.num_heads
                parameter_quantity += c * c * num_heads
            elif 'SwinTEB' in module.__class__.__name__:
                if len(module.input_shape) == 3:
                    # c, h, w = module.input_shape # c, h, w
                    _, N, c = module.input_shape
                    # N = h * w
                elif len(module.input_shape) == 2:
                    N = module.input_shape[0]
                num_heads = module.num_heads
                # hh = nH * h WindowAttention只减少了flops并没有减少显存占用，因此参数量按照图像大小算
                parameter_quantity += N * N * num_heads
                print(parameter_quantity, N, module.input_shape)
            elif 'MSA' == module.__class__.__name__:
                # L, B, D
                # if hasattr(module, '__name__'):
                #     print('model.body.decoder.layers.0.self_attn')
                # print(module.__name__, module.input_shape)
                module.input_shape = torch.from_numpy(
                    np.array(input[0].permute(1, 2, 0).size()[1:], dtype=np.int32))
                c, L = module.input_shape
                num_heads = module.num_heads
                parameter_quantity += L * L * num_heads
                # print(L, c)
            elif 'MSA_BNC' == module.__class__.__name__:
                # B, L, C
                module.input_shape = torch.from_numpy(
                    np.array(input[0].permute(0, 2, 1).size()[1:], dtype=np.int32))
                c, L = module.input_shape
                num_heads = module.num_heads
                parameter_quantity += L * L * num_heads
                # print(L, c)
            elif 'sGCN' == module.__class__.__name__:
                module.input_shape = torch.from_numpy(
                    np.array(input[0][0].permute(0, 2, 1).size(), dtype=np.int32))
                c, H, W = module.input_shape
                c = c // 2
                parameter_quantity += c * c
            elif 'cGCN' == module.__class__.__name__:
                module.input_shape = torch.from_numpy(
                    np.array(input[0][0].permute(0, 2, 1).size(), dtype=np.int32))
                c, H, W = module.input_shape
                c = c // 2
                parameter_quantity += c * c // 2
            else:
                for s in output.size()[1:]:
                    inference_memory *= s
                # memory += parameters_number  # exclude parameter memory
            for name, p in module._parameters.items():
                parameter_quantity += (0 if p is None else torch.numel(p.data))
            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.long))

            inference_memory = inference_memory * 4 / (1024 ** 2)  # shown as MB unit
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            if len(input) == 1:
                madd = compute_madd(module, input[0], output)
                flops = compute_flops(module, input[0], output)
                Memory = compute_memory(module, input[0], output)
            elif len(input) > 1:
                madd = compute_madd(module, input, output)
                flops = compute_flops(module, input, output)
                Memory = compute_memory(module, input, output)
            else:  # error
                madd = 0
                flops = 0
                Memory = (0, 0)
            module.MAdd = torch.from_numpy(
                np.array([madd], dtype=np.int64))
            module.Flops = torch.from_numpy(
                np.array([flops], dtype=np.int64))
            Memory = np.array(Memory, dtype=np.int64) * itemsize
            module.Memory = torch.from_numpy(Memory)

            return output

        leaf_modules = self.leaf_modules
        # for m in self._model.modules():
        #     print(m.__class__)

        for name, module in self._model.named_modules():
            if len(list(module.children())) == 0:
                module.name = name
                leaf_modules.append((name, module))
                if module.__class__ not in self._origin_call:
                    # 只记录一类与具体实例无关的__call__
                    self._origin_call[module.__class__] = module.__class__.__call__
                    module.__class__.__call__ = wrap_call
            elif name != '' and len(list(module.children())) > 0 and any([L in module.__class__.__name__ for L in self.debug_layers]):
                #name in self.debug_layers:# module.__class__.__name__  in self.debug_layers
                # if module.__class__.__name__ in self.debug_layers:
                #     print("111")
                leaf_modules.append((name, module))
                if module.__class__ not in self._origin_call:
                    self._origin_call[module.__class__] = module.__class__.__call__
                    module.__class__.__call__ = wrap_call
                    print(name, module.__class__.__name__)

        # for module in self._model.modules():
        #     if len(list(module.children())) == 0 and module.__class__ not in self._origin_call:
        #         self.hooks.append(module.register_forward_hook(wrap_call))

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._sub_module_call_hook()

    def clear_hooks(self) -> None:
        """Clear model hooks"""

        # for handle in self.hook_handles:
        #     handle.pop()
        def unwarp_calls(module):
            if module.__class__ in self._origin_call:
                module.__class__.__call__ = self._origin_call[module.__class__]
                # module.__delattr__('__name__')

        calls = list(map(unwarp_calls, self._model.modules()))
        del calls
        # for module in self._model.modules():
        #     if module.__class__ in self._origin_call:
        #         module.__class__.__call__ = self._origin_call[module.__class__]

    # @staticmethod
    # def _retrieve_leaf_modules(model):
    #     leaf_modules = []
    #     for name, m in model.named_modules():
    #         if len(list(m.children())) == 0:
    #             leaf_modules.append((name, m))
    #     return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self.leaf_modules)
        # return OrderedDict(self._retrieve_leaf_modules(self._model))

    def debug_partial_layer(self, target_keys):
        target_layers = []
        submodule_name = dict(list(self._model.named_modules())[1:]).keys()
        for t in target_keys:
            for name in submodule_name:
                if t in name:
                    target_layers.append(name)

        return target_layers
