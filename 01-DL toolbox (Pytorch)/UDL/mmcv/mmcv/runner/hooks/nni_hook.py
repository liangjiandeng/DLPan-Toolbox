from .hook import HOOKS, Hook

@HOOKS.register_module()
class NNIHook(Hook):

    def before_run(self, runner):
        if runner.opt_cfg['mode'] == "nni":
            import nni
            runner.logger = None
            self.nni = nni

    def after_train_epoch(self, runner):
        opt_cfg = runner.opt_cfg
        if opt_cfg['mode'] == 'nni':
            # stats = runner.outputs['log_vars']
            stats = runner.metrics
            if len(runner.workflow) == 1 and runner.epoch == runner.max_epochs:
                self.nni.report_final_result({name: value for name, value in stats.items() if opt_cfg['metrics'] in name})
            else:
                print("report_intermediate_result")
                metrics = {name: value for name, value in stats.items() if opt_cfg['metrics'] in name}
                self.nni.report_intermediate_result(metrics['loss'])


    def after_train_iter(self, runner):
        ...

    def before_val_iter(self, runner):
        ...

    def after_val_iter(self, runner):
        ...

    def after_val_epoch(self, runner):
        opt_cfg = runner.opt_cfg
        if opt_cfg['mode'] == 'nni':
            stats = runner.outputs
            if len(runner.workflow) != 1 and runner.epoch == runner.max_epochs:
                self.nni.report_final_result({name: value for name, value in stats.items() if opt_cfg['metrics'] in name}['loss'])
            else:
                self.nni.report_intermediate_result(
                    {name: value for name, value in stats.items() if opt_cfg['metrics'] in name})