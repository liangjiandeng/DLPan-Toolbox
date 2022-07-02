# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from mmcv.utils.logging import print_log
from mmcv.fileio import FileClient
from ..dist_utils import allreduce_params, master_only
from .hook import HOOKS, Hook
from math import inf
import os
import re
from ..checkpoint import save_checkpoint, get_best_k_model
import platform
import mmcv
import shutil

@HOOKS.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``.
            `Changed in version 1.3.16.`
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool, optional): Whether to force the last checkpoint to be
            saved regardless of interval. Default: True.
        sync_buffer (bool, optional): Whether to synchronize buffers in
            different gpus. Default: False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`

    .. warning::
        Before v1.3.16, the ``out_dir`` argument indicates the path where the
        checkpoint is stored. However, since v1.3.16, ``out_dir`` indicates the
        root directory and the final path to save checkpoint is the
        concatenation of ``out_dir`` and the last level directory of
        ``runner.work_dir``. Suppose the value of ``out_dir`` is "/path/of/A"
        and the value of ``runner.work_dir`` is "/path/of/B", then the final
        path will be "/path/of/A/B".
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 sync_buffer=False,
                 file_client_args=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

        print_log((f'Checkpoints will be saved to {self.out_dir} by '
                   f'{self.file_client.name}.'), logger=runner.logger)

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        if 'create_symlink' in self.args:
            if self.args[
                'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    ('create_symlink is set as True by the user but is changed'
                     'to be False because creating symbolic link is not '
                     f'allowed in {self.file_client.name}'))
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            print_log(
                f'Saving checkpoint at {runner.epoch + 1} epochs', logger=runner.logger)
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            print_log(
                f'Saving checkpoint at {runner.iter + 1} iterations', logger=runner.logger)
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)



@HOOKS.register_module()
class ModelCheckpoint(Hook):

    rule_map = {'greater': lambda x, y: x >= y, 'less': lambda x, y: x <= y}
    indicator_rule_map = {'greater': lambda x, y: max(x, y), 'less': lambda x, y: min(x, y)}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc', 'psnr', 'ssim', 'q'
    ]
    _default_best_prec1 = {'greater': -inf, 'less': inf}
    _default_less_keys = ['loss', 'sam', 'ergas']

    def __init__(self, indicator: str, formatter_filename="model_best_{epoch},{best_metric}", print_freq=1, save_top_k: int=1,
                 greater_keys=None, less_keys=None, best_prec1=None, best_epoch=0, sync_buffer=False):
        '''
        Args:
            save_interval:
            save_top_k: ``save_top_k == k``,
                        if ``save_top_k == 0``, no models are saved.
                        if ``save_top_k == -1``, all models are saved.
                        Please note that the monitors are checked every ``every_n_epochs`` epochs.
            Returns:
        '''
        self.best_epoch = best_epoch
        self.print_freq = print_freq
        self.save_top_k = save_top_k
        self.sync_buffer = sync_buffer
        self.indicator = 'top-1' if indicator == 'top' else indicator
        self.formatter_filename = formatter_filename

        # indicator_lc = indicator.lower()

        if greater_keys is None:
            greater_keys = ModelCheckpoint._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys,)
            # assert is_seq_of(greater_keys, str)
            greater_keys = [key.lower() for key in greater_keys]

        if less_keys is None:
            less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys,)
            # assert is_seq_of(less_keys, str)
            less_keys = [key.lower() for key in less_keys]

        if indicator in greater_keys:
            rule = 'greater'
        elif indicator in less_keys:
            rule = 'less'
        elif any(key in indicator for key in greater_keys):
            rule = 'greater'
        elif any(key in indicator for key in less_keys):
            rule = 'less'
        else:
            raise ValueError(f'Cannot infer the rule for key '
                             f'{indicator}, thus a specific rule '
                             f'must be specified.')
        self.best_prec1 = self._default_best_prec1[rule] if best_prec1 is None else best_prec1
        self.compare_func = self.rule_map[rule]
        self.indicator_func = self.indicator_rule_map[rule]
        self.rule = rule

    def before_run(self, runner):
        self.save_model_path = runner.work_dir
        self.ckpt = os.path.join(self.save_model_path, 'checkpoint')
        os.makedirs(self.save_model_path, exist_ok=True)
        print_log(f'Checkpoints will be saved to {self.save_model_path}', logger=runner.logger)


    def earlyStopping(self, avg_grad_norm):

        if avg_grad_norm > 100:
            return True


    def after_train_epoch(self, runner):
        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        metrics = runner.metrics# metrics = {k: meter.avg for k, meter in runner.log_buffer.meters.items()}
        runner.earlyStop = self.earlyStopping(metrics.get('grad_norm', 0))
        self.save_checkpoint(runner, metrics)

        # print_log(' * Best training metrics so far@ {best_metric} in epoch {best_epoch}'.format(
        #     best_metric=metrics['best_metric'], best_epoch=metrics['best_epoch']), logger=runner.logger)

    def _save_checkpoint(self, meta, out_dir, filename, is_best, create_symlink=True):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        # meta.update(epoch=meta.pop('epoch') + 1, iter=meta.pop('iter'))
        filepath = os.path.join(out_dir, filename)
        # save_checkpoint(meta.pop('model'), filepath, optimizer=meta.pop('optimizer'), meta=meta)
        save_checkpoint(filepath, meta=meta)
        if create_symlink or is_best:
            dst_file = os.path.join(out_dir, 'model_best_.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    @master_only
    def save_checkpoint(self, runner, metrics):
        flag = False
        if not hasattr(runner.model, 'train') and isinstance(runner.model.model, dict):
            flag = True
            stats = {}
            for k, m in runner.model.model.items():
                stats[k] = {
                    'epoch': runner.epoch,
                    'iter': runner.iter,
                    'model': m,
                    'best_metric': {name: value for name, value in metrics.items() if
                                    name not in ['grad_norm', 'lr', 'time', 'data_time']},
                    # 保存多个metric的数值,  实际比较的时候还是只有一个
                    'loss': metrics['loss'],
                    'best_epoch': runner._epoch,
                    'optimizer': runner.optimizer[k]
                }
                runner.metrics.update(
                    {'best_metric': {k: stats[k]['best_metric']}, 'best_epoch': {k: stats[k]['best_epoch']}})
        else:
            stats = {
                'epoch': runner.epoch,
                'iter': runner.iter,
                'model': runner.model,
                'best_metric': {name: value for name, value in metrics.items() if
                                name not in ['grad_norm', 'lr', 'time', 'data_time']},
                # 保存多个metric的数值,  实际比较的时候还是只有一个
                'loss': metrics['loss'],
                'best_epoch': runner._epoch,
                'optimizer': runner.optimizer
            }
            runner.metrics.update(best_metric=stats['best_metric'], best_epoch=stats['best_epoch'])

        new_best_k_model_flag = []
        indicator = self.indicator
        save_top_k = self.save_top_k
        # stats 应当是{epoch: X, score: Y} -> [epoch, score]
        assert isinstance(stats, dict), print(f"stats in model_checkpoint should be dict but be {type(stats)}")
        # stats = list(stats.values())
        best_k_model, _ = get_best_k_model(self.save_model_path + "/checkpoint", indicator)

        # print(best_k_model)
        if save_top_k < 0:
            raise ValueError(f"Invalid value for save_top_k={save_top_k}. Must be >= 0")
        if save_top_k == 0:
            stats['best_metric'] = self._default_best_prec1[self.rule]
            stats['best_epoch'] = 0
            self._save_checkpoint(stats, self.save_model_path, is_best=False, filename=f"{stats['epoch']}.pth.tar")

        if save_top_k >= 1:
            # self.best_prec1 = self.indicator_func(self.best_prec1, stats[self.indicator])
            if len(best_k_model) >= save_top_k:
                # reverse=True, 降序, default： False
                # 使用索引去对best_k_model进行排序,best_k_model应是列表，才能返回索引
                best_k_model.append([stats['epoch'], stats['best_metric'], None])
                sortedIndex_best_k_model = sorted(range(len(best_k_model)),
                                                  key=lambda k: float(best_k_model[k][1][indicator]),
                                                  reverse=self.rule == "less")
                # print(sortedIndex_best_k_model)
                new_best_k_model_flag = [
                    not self.compare_func(float(query_score[indicator]), stats['best_metric'][indicator]) for
                    _, query_score, _ in best_k_model]
                # print(new_best_k_model_flag)

                # ckpt_stats = [] # {}
                # key会冲突导致popitem出错
                # ckpt_stats[str(stats['epoch'])] = stats[indicator]
                ckpt_stats = [stats['epoch'], stats['best_metric']]

                for index in sortedIndex_best_k_model:
                    if new_best_k_model_flag[index]:
                        # top_k_count += 1
                        # best_k_model[indicator][index] = stats[indicator]
                        # best_k_model['epoch'][index] = stats['epoch']
                        # best_k_model.pop(str(index))
                        # best_k_model.update(ckpt_stats)
                        # best_k_model[index] = list(ckpt_stats.popitem())
                        fname = self.save_model_path + "/" + best_k_model[index][2]
                        ckpt_stats.append(None)
                        best_k_model[index] = ckpt_stats

                        if os.path.isfile(fname):
                            os.remove(fname)
                        break
                stats['best_epoch'], stats['best_metric'] = best_k_model[sortedIndex_best_k_model[-1]][:2]
                best_k_model = best_k_model[:-1]
                # best_k_model = [{'epoch': k, 'score': v} for k, v in best_k_model.items()]
                best_k_model = [{'epoch': epoch, 'best_metric': score} for (epoch, score, _) in best_k_model]
                with open(self.ckpt, 'w') as f:
                    outs = [self.formatter_filename.format(**line) + "\n" for line in best_k_model]
                    f.writelines(outs)
            else:
                if not flag:
                    with open(self.ckpt, 'a') as f:
                        outs = self.formatter_filename.format(**stats) + "\n"
                        f.writelines(outs)
                # 训练初期，不满topk时候, 模型是否保存下来
                # if save_top_k == 1:
                # if len(best_k_model) < save_top_k:
                #     new_best_k_model_flag = [True]

            is_best = any(new_best_k_model_flag)
            if runner.epoch % self.print_freq == 0 or is_best:
                self._save_checkpoint(
                        stats, out_dir=self.save_model_path, is_best=is_best, filename=f"{runner.epoch}.pth.tar")

                if not flag:
                    print_log(' * Best training metrics so far@ {best_metric} in epoch {best_epoch}'.format(
                        best_metric=stats['best_metric'], best_epoch=stats['best_epoch']), logger=runner.logger
                    )

            return stats

    def after_train_iter(self, runner):
        if hasattr(runner.model, 'train'):
            if type(runner.model.module.model).__name__ == 'INN':
                runner.model.module.model.free()
        else:
            if isinstance(runner.model.model, dict):
                runner.model.model['PAN2MS'].module.free()
    # raise NotImplementedError("after_train_iter is not implemented by ModelCheckpoint (customed)")






