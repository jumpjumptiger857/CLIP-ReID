# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,                  # 学习率衰减的步数列表（如[10000, 20000]）
            gamma=0.1,                   # 衰减系数（默认0.1，即降为1/10）
            warmup_factor=1.0 / 3,       # 热身初始学习率系数（默认1/3）
            warmup_iters=500,            # 热身步数（默认500步）
            warmup_method="linear",
            last_epoch=-1,               # 上一次的训练步数（默认-1，从头开始）
    ):
        # 校验1：milestones必须是递增的整数列表（如[5000, 10000]，不能是[10000,5000]）
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        # 校验2：热身方式只能是constant或linear
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        # 保存核心参数
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        # 调用父类_LRScheduler的初始化（必须）
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
