from src import *
from typing import Optional, Union, Type, Callable, Dict, Any

import torch
import lightning.pytorch as pl
from lightning.pytorch import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback, ArgsType, OptimizerCallable, LRSchedulerCallable


class MyCLI(LightningCLI):
    def __init__(self, 
            model_class: Optional[Union[Type[pl.LightningModule], Callable[..., LightningModule]]] = None, 
            datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None, 
            save_config_callback: Optional[Type[SaveConfigCallback]] = ..., 
            save_config_kwargs: Optional[Dict[str, Any]] = None, 
            trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = ..., 
            trainer_defaults: Optional[Dict[str, Any]] = None, 
            seed_everything_default: Union[bool, int] = True, 
            parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None, 
            subclass_mode_model: bool = False, 
            subclass_mode_data: bool = False, 
            args: ArgsType = None, run: bool = True, 
            auto_configure_optimizers: bool = True) -> None:
        super().__init__(
            model_class, datamodule_class, save_config_callback, save_config_kwargs, trainer_class, trainer_defaults,
            seed_everything_default, parser_kwargs, subclass_mode_model, subclass_mode_data, args, run, auto_configure_optimizers
        )


def main_cli():
    cli = LightningCLI()

if __name__ == '__main__':
    main_cli()