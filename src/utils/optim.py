import torch
from torch.optim.lr_scheduler import LRScheduler


def setup_optimization(parameters, optimizer_name, optimizer_args, scheduler_args, **args):
    # -------------- instantiate optimizer and scheduler
    optimizer_args = {k: float(v) for k, v in optimizer_args.items()}
    optimizer = getattr(torch.optim, optimizer_name)(parameters, **optimizer_args)
    if scheduler_args['scheduler'] == 'ReduceLROnPlateau':
        print("Instantiating ReduceLROnPlateau scheduler.")
        scheduler = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_args['scheduler_mode'],
                                                                factor=float(scheduler_args['decay_rate']),
                                                                patience=scheduler_args['patience'],
                                                                verbose=True), scheduler_args['scheduler']]
    elif scheduler_args['scheduler'] == 'StepLR':
        scheduler = [torch.optim.lr_scheduler.StepLR(optimizer, float(scheduler_args['decay_steps']), gamma=float(scheduler_args['decay_rate'])),
                     scheduler_args['scheduler']]

    elif scheduler_args['scheduler'] == 'WarmupLRScheduler':
        scheduler = [WarmupLRScheduler(optimizer, warmup_steps=scheduler_args['warmup_steps']), scheduler_args['scheduler']]

    elif args['scheduler'] is None:
        scheduler = [None, None]
    else:
        raise NotImplementedError('Scheduler {} is not currently supported.'.format(args['scheduler']))
    return optimizer, scheduler


class WarmupLRScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps=10000, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [
                base_lr * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs