import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import (CALLBACK_REGISTRY,
                                             LR_SCHEDULER_REGISTRY,
                                             OPTIMIZER_REGISTRY, LightningCLI)

# 自定义callbacks
import helpers.callbacks as custom_callbacks
# 自定义lr策略
import helpers.lr_scheduler as custom_lr_scheduler

# 不知道下面这两句是干啥的
# 注册所有满足custom_callbacks的子类
CALLBACK_REGISTRY.register_classes(
    pl.callbacks, pl.callbacks.Callback, custom_callbacks)
# 注册所有满足custom_lr_scheduler的子类 
LR_SCHEDULER_REGISTRY.register_classes(
    torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler, custom_lr_scheduler)


class ConditioningLightningCLI(LightningCLI):
    # OPTIMIZER_REGISTRY.classes就是获取被注册过的类
    # nested_key是配置文件中最上层的命名空间的名字
    # 向parse传递额外的参数
    # link到model的optimizer_init和
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes, nested_key="optimizer", link_to="model.init_args.optimizer_init")
        parser.add_lr_scheduler_args(
            LR_SCHEDULER_REGISTRY.classes, nested_key="lr_scheduler", link_to="model.init_args.lr_scheduler_init")
