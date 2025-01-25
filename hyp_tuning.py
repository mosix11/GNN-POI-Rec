import torch
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.dataset import FoursquareNYC
from src.models import Baseline, HMT_GRN, HMT_GRN_V2

import os
from pathlib import Path
from functools import partial

# HYP_SEARCH_SPACE = {
#     "user_emb_dim": tune.choice([128, 256, 512]),
#     "poi_emb_dim": tune.choice([256, 512, 768]),
#     "poi_cat_emb_dim": tune.choice([128, 256]),
#     "gh_emb_dim": tune.choice([128, 256, 512]),
#     "ts_emb_dim": tune.choice([128, 256]),
#     "hidden_dim": tune.choice([256, 512, 1024]),
    # "emb_switch": tune.choice([
    #     [True, True, True, True],
    #     [True, True, True, False],
    #     [True, True, False, True],
    #     [True, True, False, False]
    # ]),
#     "num_lstm_layers": tune.choice([1, 2]),
#     "lstm_dropout": tune.choice([0.0, 0.5]),
#     "emb_dropout": tune.choice([0.0, 0.5, 0.9]),
#     "GAT_dropout": tune.choice([0.0, 0.5]),
    # "task_loss_coefficients": tune.choice([
    #     [1., 1., 1., 1.],
    #     [1., 0.5, 0.5, 0.5],
    #     [1., 0., 0. , 0.],
    #     [1., 0., 0.5, 0]
    # ]),
#     "optim_lr": tune.loguniform(1e-5, 5e-4),
#     "optim_type": tune.choice(['adam', 'adamw'])
# }


def objective(trial:optuna.Trial):
    user_emb_dim = trial.suggest_categorical("user_emb_dim", [128, 256, 512])
    poi_emb_dim = trial.suggest_categorical("poi_emb_dim", [256, 512, 768])
    poi_cat_emb_dim = trial.suggest_categorical("poi_cat_emb_dim", [128, 256])
    gh_emb_dim = trial.suggest_categorical("gh_emb_dim", [128, 256, 512])
    ts_emb_dim = trial.suggest_categorical("ts_emb_dim", [128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    emb_switch = trial.suggest_categorical("emb_switch", [
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, True],
        [True, True, False, False]
    ])
    num_lstm_layers = trial.suggest_categorical("num_lstm_layers", [1, 2])
    lstm_dropout = trial.suggest_categorical("lstm_dropout", [0.0, 0.5])
    emb_dropout = trial.suggest_categorical("emb_dropout", [0.0, 0.5, 0.9])
    GAT_dropout = trial.suggest_categorical("GAT_dropout", [0.0, 0.5])
    task_loss_coefficients = trial.suggest_categorical("task_loss_coefficients", [
        [1., 1., 1., 1.],
        [1., 0.5, 0.5, 0.5],
        [1., 0., 0. , 0.],
        [1., 0., 0.5, 0]
    ])
    optim_lr = trial.suggest_categorical


def train_func(config):
    dataset = FoursquareNYC(
            batch_size=8,
            spatial_graph_self_loop=True,
            temporal_graph_self_loop=True,
            temporal_graph_jaccard_mult_set=False,
            temporal_graph_jaccard_sim_tsh=0.5,
        )
    Model_factory = partial(HMT_GRN_V2, dataset=dataset)
    model = Model_factory(**config)


    trainer = pl.Trainer(
        max_epochs=100,
        devices="auto",
        accelerator="auto",
        callbacks=[
            TuneReportCallback(
                {"loss": "Val/POI Loss", "ACC@1": "Val/Acc@1", "ACC@5": "Val/Acc@5"},
                on="validation_end",
            )
        ],
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dataset)
    
if __name__ == '__main__':
    
    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=3,
        reduction_factor=2
        )
    reporter = CLIReporter(
        parameter_columns=["poi_emb_dim", "emb_switch", "task_loss_coefficients", "optim_lr"],
        metric_columns=["loss", "ACC@1", "ACC@5"],
    )
    
    resources_per_trial = {"cpu": 4, "gpu": 1}
    
    tuner = tune.Tuner(
        tune.with_resources(train_func, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=10,
        ),
        run_config=train.RunConfig(
            name="GRN",
            progress_reporter=reporter,
        ),
        param_space=HYP_SEARCH_SPACE,
    )
    
    results = tuner.fit()
    best_trial = results.get_best_result(metric="Val/POI Loss", mode="min")
    print(f"Best trial final metrics: {best_trial.metrics}")
    print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(
    #     best_trial.metrics["Val/POI Loss"]))
    # print(f"Best trial final metrics: {best_trial.metrics}")