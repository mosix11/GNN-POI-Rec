import torch
import pytorch_lightning as pl

from ray import train, tune
from ray.tune import CLIReporter
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from files.src.models import baseline
from src.dataset import FoursquareNYC
from src.models import HMT_GRN, HMT_GRN_V2
from src.trainer import CustomTrainer

import os
from pathlib import Path
from functools import partial

HYP_SEARCH_SPACE = {
    "user_emb_dim": tune.choice([256, 512]),
    "poi_emb_dim": tune.choice([512, 768]),
    "gh_emb_dim": tune.choice([128, 256]),
    "ts_emb_dim": tune.choice([128, 256]),
    "hidden_dim": tune.choice([512, 768, 1024]),
    "emb_switch": tune.choice([
        # [True, True, True, True],
        # [True, True, True, False],
        [True, True, False, True],
        # [True, True, False, False]
    ]),
    "emb_dropout": tune.choice([0.9]),
    "GAT_dropout": tune.choice([0.0, 0.5]),
    "task_loss_coefficients": tune.choice([
        [1., 1., 1., 1.],
        [1., 0., 0. , 0.],
        [1., 0., 0.5, 0]
    ]),
    "optim_lr": tune.loguniform(1e-5, 5e-4),
}


class ModifiedASHAScheduler(ASHAScheduler):
    def on_trial_add(self, tune_controller, trial):
        trial.config["trial_name"] = str(trial)
        super().on_trial_add(tune_controller, trial)

def objective(config, dataset:FoursquareNYC, outputs_dir:Path):
    
    trial_name = config['trial_name']
    config.pop('trial_name')
    model = HMT_GRN_V2(dataset=dataset, **config)
    
    trainer = CustomTrainer(max_epochs=14, outputs_dir=outputs_dir, ray_tuner=train, trial_name=trial_name)
    
    # if train.get_checkpoint():
    #     loaded_checkpoint = train.get_checkpoint()
    #     with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
    #         checkpoint = torch.load(
    #             os.path.join(loaded_checkpoint_dir, "ckp.pt")
    #         )
    #         trainer.fit(model, dataset, checkpoint)
    # else:
    trainer.fit(model, dataset)
            


if __name__ == "__main__":

    outputs_dir = Path("raytune").absolute()
    outputs_dir.mkdir(exist_ok=True)
    
    dataset = FoursquareNYC(
        batch_size=8,
    )
    dataset.setup('fit')
    dataset.setup('test')

    scheduler = ModifiedASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2,
        )
    
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(objective, dataset=dataset, outputs_dir=outputs_dir),
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=200,
        ),
        
        param_space=HYP_SEARCH_SPACE,
    )
    results = tuner.fit()
    
    trials_df = results.get_dataframe()
    csv_file_path = outputs_dir / Path("tuning_results.csv")
    trials_df.to_csv(csv_file_path, index=False)
    
    
    # Extract the best trial run from the search.
    best_trial_loss = results.get_best_result('loss', 'min', 'last')
    best_trial_acc = results.get_best_result('accuracy', 'max')

    print("Best trial config: {}".format(best_trial_loss.config))
    print("Best trial final validation loss: {}".format(
        best_trial_loss.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial_loss.metrics["accuracy"]))
    
    print('\n\n', best_trial_acc)