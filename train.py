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
        default="HMT_GRN.yaml",
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
        action="store_true",
    )
    args = parser.parse_args()

    # cfg_path = Path('configs').joinpath(args.config)
    # if not cfg_path.exists(): raise RuntimeError('The specified config file was not found.')
    # with open(cfg_path, 'r') as file:
    #     cfg = yaml.full_load(file)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()

    use_amp = True

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    log_dir = outputs_dir / Path("tensorboard")
    log_dir.mkdir(exist_ok=True)

    if args.model == "baseline":
        ds = FoursquareNYC(batch_size=128)

        model = TrajLSTM(ds)

        tb_logger = TensorBoardLogger(log_dir, name="Baseline")

        trainer = Trainer(
            max_epochs=200,
            accelerator="gpu",
            logger=tb_logger,
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

        ds = FoursquareNYC(
            batch_size=8,
            # max_traj_length=128,
            spatial_graph_self_loop=True,
            temporal_graph_self_loop=True,
            temporal_graph_jaccard_mult_set=False,
            temporal_graph_jaccard_sim_tsh=0.5,
        )
        model = HMT_GRN_V2(ds)
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
