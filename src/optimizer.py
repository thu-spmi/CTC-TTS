import inspect
from abc import ABC, abstractmethod

import torch


def _build_param_groups(model, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


class OptimizerStrategy(ABC):
    def __init__(self, config, device_type):
        self.config = config
        self.device_type = device_type

    @abstractmethod
    def build(self, model):
        raise NotImplementedError


class AdamWStrategy(OptimizerStrategy):
    def build(self, model):
        training = self.config["training"]
        param_groups = _build_param_groups(model, training["weight_decay"])
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(
            param_groups,
            lr=training["learning_rate"],
            betas=(training["beta1"], training["beta2"]),
            eps=training["eps"],
            **extra_args,
        )


class AdamStrategy(OptimizerStrategy):
    def build(self, model):
        training = self.config["training"]
        param_groups = _build_param_groups(model, training["weight_decay"])
        return torch.optim.Adam(
            param_groups,
            lr=training["learning_rate"],
            betas=(training["beta1"], training["beta2"]),
            eps=training["eps"],
        )


class SGDStrategy(OptimizerStrategy):
    def build(self, model):
        training = self.config["training"]
        param_groups = _build_param_groups(model, training["weight_decay"])
        momentum = training.get("momentum", 0.9)
        nesterov = training.get("nesterov", False)
        return torch.optim.SGD(
            param_groups,
            lr=training["learning_rate"],
            momentum=momentum,
            nesterov=nesterov,
        )


def build_optimizer(config, model, device_type):
    optimizer_type = config["training"].get("optimizer", "adamw").lower()
    if optimizer_type == "adamw":
        strategy = AdamWStrategy(config, device_type)
    elif optimizer_type == "adam":
        strategy = AdamStrategy(config, device_type)
    elif optimizer_type == "sgd":
        strategy = SGDStrategy(config, device_type)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return strategy.build(model)
