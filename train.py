import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    LearningRateFinder,
)


from src.utils import misc_utils, nn_utils

import os
import sys
import argparse
import yaml
from pathlib import Path

from src.dataset import FoursquareNYC
from src.models import TrajLSTM, HMT_GRN_V2

# The entery point of the training and evaluation procedure.
# Based on the passed argument for --model the training will be done
# on the Baseline model or on the HMT-GRN model. Also for each model
# the specified configuration file name should be passed as --config.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="The model to train between [`grn`, `baseline`]",
        type=str,
        choices=["grn", "baseline"],
        default="grn",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to use for training the model.",
        type=str,
    )

    args = parser.parse_args()
    if not args.config:
        raise RuntimeError("You have to specify the training config file.")
    cfg_path = Path("configs").joinpath(args.config)
    if not cfg_path.exists():
        raise RuntimeError("The specified config file was not found.")
    with open(cfg_path, "r") as file:
        cfg = yaml.full_load(file)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.set_float32_matmul_precision("high")

    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()

    use_amp = True

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    log_dir = outputs_dir / Path("tensorboard")
    log_dir.mkdir(exist_ok=True)

    if args.model == "baseline":
        ds = FoursquareNYC(**cfg["dataset"])

        model = TrajLSTM(ds, **cfg["model"])

        tb_logger = TensorBoardLogger(log_dir, name="Baseline")

        trainer = Trainer(
            max_epochs=200,
            accelerator="gpu",
            logger=tb_logger,
            enable_checkpointing=False,
            strategy="auto",
            callbacks=[
                EarlyStopping(
                    monitor="Val/Loss", patience=20, min_delta=0.001, mode="min"
                ),
            ],
        )

        trainer.fit(model, datamodule=ds)
        trainer.test(model, datamodule=ds)
    elif args.model == "grn":

        ds = FoursquareNYC(**cfg["dataset"])
        model = HMT_GRN_V2(ds, **cfg["model"])
        tb_logger = TensorBoardLogger(log_dir, name="HMT-GRN")

        trainer = Trainer(
            max_epochs=200,
            accelerator="gpu",
            logger=tb_logger,
            strategy="auto",
            callbacks=[
                EarlyStopping(
                    monitor="Val/Loss", patience=25, min_delta=0.001, mode="min"
                ),
            ],
        )
        trainer.fit(model, datamodule=ds)
        trainer.test(model, datamodule=ds)

    else:
        raise RuntimeError("Invalid model type. Valid choices are [`grn`, `baseline`]")
