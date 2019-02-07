import math
import torch.optim as optim


class GeomRampUpLinearDecayLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer,
                 n_iters, n_rampup_iters, n_iters_before_decay,
                 rampup_gamma, middle_coeff=1, last_epoch=-1):
        self.n_iters = n_iters
        self.n_iters_before_decay = n_iters_before_decay
        self.n_rampup_iters = n_rampup_iters
        self.start_decay_from = n_iters - n_iters_before_decay
        self.rampup_gamma = rampup_gamma
        self.middle_coeff = middle_coeff
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        first_lrs = [
            base_lr * math.pow(self.rampup_gamma, min(self.n_rampup_iters, self.last_epoch))
            for base_lr in self.base_lrs
        ]
        if self.last_epoch < self.n_rampup_iters:
            return first_lrs

        middle_lrs = [
            base_lr * 1.0/math.sqrt(
                1 + self.middle_coeff * min(self.start_decay_from, self.last_epoch - self.n_rampup_iters))
            for base_lr in first_lrs
        ]
        if self.last_epoch <= self.start_decay_from:
            return middle_lrs

        n_iters_before_finish = self.n_iters - self.last_epoch
        slope = n_iters_before_finish / self.n_iters_before_decay
        last_lrs = [
            base_lr * slope
            for base_lr in middle_lrs
        ]
        return last_lrs

