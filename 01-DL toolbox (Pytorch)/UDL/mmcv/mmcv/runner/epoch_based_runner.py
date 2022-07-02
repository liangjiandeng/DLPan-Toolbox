# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform
import shutil
import time
import warnings
import time
import datetime
import torch
import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info
from mmcv.utils.logging import print_log
from .record import MetricLogger, get_grad_norm


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                        **kwargs)

            # if not isinstance(self.model, dict):
            #     outputs = self.model.train_step(data_batch, self.optimizer,
            #                                     **kwargs)
            # else:
            #     outputs = {}
            #     for name in self.model.keys():
            #         outputs.update(self.model[name].train_step(data_batch, self.optimizer,
            #                                         **kwargs))

        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update_dict(outputs['log_vars'])
        # {'loss': loss, 'log_vars': {'loss': loss, 'metric_1': ..., 'metric_2': ....} }
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        if hasattr(self.model, 'train'):
            self.model.train()
        elif isinstance(self.model.model, dict):
            for name in self.model.model.keys():
                self.model.model[name].train()
        else:
            self.model.model.train()
        # if not isinstance(self.model, dict):
        #     self.model.train()
        # else:
        #     for name in self.model.keys():
        #         self.model[name].train()

        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.metrics = {k: meter.avg for k, meter in self.log_buffer.meters.items()}
        self.call_hook('after_train_epoch')
        self._epoch += 1

    def simple_train(self, data_loader, **kwargs):
        optimizer = self.optimizer
        accumulated_step = self.opt_cfg.get('accumulated_step', 1)
        clip_max_norm = self.opt_cfg.get('clip_max_norm', 0)
        print_freq = self.opt_cfg.get('print_freq', 1)
        nni = self.opt_cfg.get('nni', None)
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        # metric_logger = MetricLogger(delimiter="  ", dist_print=0, logger=self.logger)
        header = 'Epoch: [{}]'.format(self._epoch)
        print_freq = len(data_loader) if print_freq <= 0 else print_freq
        metric_logger = self.log_buffer
        for data_batch, idx in metric_logger.log_every(data_loader, print_freq, header):
            self._inner_iter = idx
            self.run_iter(data_batch, train_mode=True, **kwargs)
            losses = self.outputs['loss'] / accumulated_step
            losses.backward()
            if clip_max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_max_norm)
            else:
                grad_norm = get_grad_norm(self.model.parameters())
            if idx % accumulated_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_norm)
            metric_logger.update_dict(self.outputs['log_vars'])
            self._iter += 1


        self.metrics = {k: meter.avg for k, meter in metric_logger.meters.items()}
        self.call_hook('after_train_epoch')
        metric_logger.clear()
        self._epoch += 1
        if nni is not None:
            nni.report_intermediate_result(
                {name: value for name, value in self.metrics.items() if self.opt_cfg.metrics in name})

    @torch.no_grad()
    def simple_val(self, data_loader, **kwargs):
        # 用IterBasedRunner是否会更统一？
        # 如果要更进一步整合，应该变成eval_hook,但这是一个simple case
        self.model.eval()
        self.mode = 'val'
        opt_cfg = self.opt_cfg
        save_fmt = opt_cfg['save_fmt']
        # metric_logger = MetricLogger(dist_print=0, delimiter="  ", logger=self.logger)
        metric_logger = self.log_buffer
        header = 'TestEpoch: [{0}]'.format(self.epoch - 1)
        save_dir = os.path.join(self.work_dir, f"{opt_cfg['dataset']}")
        if save_fmt and self._epoch == 1:
            os.makedirs(save_dir, exist_ok=True)
        for batch, idx in metric_logger.log_every(data_loader, 1, header):
            metrics = self.model.val_step(batch, save_dir,
                                          idx=idx, save_fmt=save_fmt, filename=batch.get('filename', None))
            # self.run_iter()
            metric_logger.update_dict(metrics)
        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        if opt_cfg['mode'] == 'nni':
            self.nni.report_final_result({name: value for name, value in stats.items() if opt_cfg['metrics'] in name})
        # 仅进行验证时触发，结束while
        metric_logger.clear()
        if not self.flag:
            self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif isinstance(self.model.model, dict):
            for name in self.model.model.keys():
                self.model.model[name].eval()
        else:
            self.model.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        tic = time.time()
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False, idx=i,
                          img_range=self.opt_cfg['img_range'],
                          save_fmt=self.opt_cfg['save_fmt'], filename=data_batch.get('filename', [None])[0], save_dir=self.save_dir)
            self.call_hook('after_val_iter')
        print("test time:", time.time() - tic)
        self.call_hook('after_val_epoch')
        if self.opt_cfg['eval']:
            self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, dict)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow), print_log(f"{len(data_loaders)} == {len(workflow)}")
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')
        self.flag = any('train' in mode for mode, _ in workflow)
        self.workflow = workflow
        self.data_length = 1
        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[mode])
                self.data_length = len(data_loaders[mode])
                break


        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        print_log(f'Start running, host: {get_host_info()}, work_dir: {work_dir}',
                  logger=self.logger)
        print_log(f'Hooks will be executed in the following order:\n{self.get_hook_info()}',
                  logger=self.logger)
        print_log(f'workflow: {workflow}, max: {self._max_epochs} epochs',
                  logger=self.logger)
        self.call_hook('before_run')
        tic = time.time()
        print_freq = self.opt_cfg.get('print_freq', 1)
        # from 1 to self._max_epochs, not from 0
        while self.epoch <= self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for epoch in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[mode], **kwargs)
            if self.earlyStop:
                print_log("model train has diverged, python will stop training", logger=self.logger)
                break
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        total_time = time.time() - tic
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_log('Training time {}'.format(total_time_str), logger=self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = os.path.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = os.path.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)
