import torch
import pytorch_lightning as pl
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from src.dataset import FoursquareNYC
from src.models import Baseline, HMT_GRN, HMT_GRN_V2

import os
from pathlib import Path
from functools import partial

HYP_SEARCH_SPACE = {
    "user_emb_dim": tune.choice([128, 256, 512]),
    "poi_emb_dim": tune.choice([256, 512, 768]),
    "poi_cat_emb_dim": tune.choice([128, 256]),
    "gh_emb_dim": tune.choice([128, 256, 512]),
    "ts_emb_dim": tune.choice([128, 256]),
    "hidden_dim": tune.choice([256, 512, 1024]),
    "emb_switch": tune.choice([
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, True],
        [True, True, False, False]
    ]),
    "num_lstm_layers": tune.choice([1, 2]),
    "lstm_dropout": tune.choice([0.0, 0.5]),
    "emb_dropout": tune.choice([0.0, 0.5, 0.9]),
    "GAT_dropout": tune.choice([0.0, 0.5]),
    "task_loss_coefficients": tune.choice([
        [1., 1., 1., 1.],
        [1., 0.5, 0.5, 0.5],
        [1., 0., 0. , 0.],
        [1., 0., 0.5, 0]
    ]),
    "optim_lr": tune.loguniform(1e-5, 5e-4),
    "optim_type": tune.choice(['adam', 'adamw'])
}

def train_func(config):
    dataset = FoursquareNYC(batch_size=8)
    Model_factory = partial(HMT_GRN_V2, dataset=dataset)
    model = Model_factory(**config)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dataset)
    
if __name__ == '__main__':
    
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=30,
        grace_period=3,
        reduction_factor=2)