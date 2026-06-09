import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_lr(self, it):
        raise NotImplementedError


class FixedLRScheduler(LRScheduler):
    def get_lr(self, it):
        return self.config["training"]["learning_rate"]


class InverseSqrtLRScheduler(LRScheduler):
    def get_lr(self, it):
        warmup_iters = self.config["training"]["warmup_iters"]
        base_lr = self.config["training"]["learning_rate"]
        min_lr = self.config["training"]["min_lr"]

        if it <= 0:
            return min_lr
        if it < warmup_iters:
            lr = min_lr + (base_lr - min_lr) * it / warmup_iters
        else:
            lr = base_lr * (warmup_iters**0.5) / (it**0.5)
        return max(lr, min_lr)


class CosineLRScheduler(LRScheduler):
    def get_lr(self, it):
        warmup_iters = self.config["training"]["warmup_iters"]
        base_lr = self.config["training"]["learning_rate"]
        min_lr = self.config["training"]["min_lr"]
        lr_decay_iters = self.config["training"]["lr_decay_iters"]

        if it < warmup_iters:
            return base_lr * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (base_lr - min_lr)


def build_lr_scheduler(config):
    if not config["training"]["decay_lr"]:
        return FixedLRScheduler(config)
    schedule = config["training"].get("lr_schedule", "cosine")
    if schedule == "inverse_sqrt":
        return InverseSqrtLRScheduler(config)
    if schedule == "cosine":
        return CosineLRScheduler(config)
    raise ValueError(f"Unknown learning rate schedule: {schedule}")
