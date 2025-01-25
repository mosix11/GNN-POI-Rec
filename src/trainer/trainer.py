
import torch
import pytorch_lightning as pl

from torch.amp import GradScaler
from torch.amp import autocast

from ray import tune
from ray.train import Checkpoint

from ..utils import nn_utils

from pathlib import Path
import time
import os

class CustomTrainer():
    
    def __init__(self,
                 max_epochs:int,
                 outputs_dir:Path,
                 ray_tuner=None,
                 trial_name=None
                 ):
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        self.max_epochs = max_epochs
        
        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(exist_ok=True)
        
        self.ray_checkpoints_dir = self.outputs_dir / Path('ray_checkpoints')
        self.ray_checkpoints_dir.mkdir(exist_ok=True)
        self.ray_tuner = ray_tuner
        self.trial_name = trial_name
        
        
    def setup_data_loaders(self, dataset):
        self.dataset = dataset
        self.train_dataloader = dataset.train_dataloader()
        self.val_dataloader = dataset.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        
    def prepare_model(self, model):
        self.model = model.to(self.gpu)
        
    def move_batch_to_device(self, batch, device):
        x, y, lens = batch
        x = [item.to(device) for item in x]
        y = [item.to(device) for item in y]
        lens = lens.to(device)
        return [x, y, lens]
    

    # def lodat_checkpoint(self, checkpoint):
        
    
    
    def fit(self, model, dataset, use_amp=False, checkpoint=None):
        
        self.use_amp = use_amp
        self.setup_data_loaders(dataset)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizers()
        
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
            self.optim.load_state_dict(checkpoint['optim_state'])
            self.epoch = checkpoint['epoch']
        else:
            self.epoch = 1
        
        self.grad_scaler = GradScaler('cuda', enabled=self.use_amp)
        
        for self.epoch in range(self.epoch, self.max_epochs+1):
            self.fit_epoch()

            
    def fit_epoch(self):
         
        self.model.train()
        epoch_start_time = time.time()
        
        self.model.poi_loss_avg.reset()
        for idx, batch in enumerate(self.train_dataloader):
            batch = self.move_batch_to_device(batch, self.gpu)
            
            with autocast('cuda', enabled=self.use_amp):
                loss = self.model.training_step(batch, idx)
                
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optim)
            self.grad_scaler.update()
        train_poi_loss = self.model.poi_loss_avg.avg
            
        print(f"Epoch {self.epoch}, POI training loss = {train_poi_loss}, Time taken: {int((time.time() - epoch_start_time)//60)}:{int((time.time() - epoch_start_time)%60)} minutes")
            

        
        if self.epoch >= 6 and self.epoch % 2 == 0:
            epoch_start_time = time.time()
        # if self.ray_tuner:
        #     ckp_dir = self.ray_checkpoints_dir / Path(self.trial_name)
        #     ckp_dir.mkdir(exist_ok=True)
        #     ckp_path = ckp_dir / Path('ckp.pt')
        #     torch.save({
        #         'model_state': self.model.state_dict(),
        #         'optim_state': self.optim.state_dict(),
        #         'epoch': self.epoch,
        #     }, ckp_path.absolute())
        #     checkpoint = Checkpoint.from_directory(ckp_dir)
            
            self.model.eval()
            self.model.poi_loss_avg.reset()
            self.model.acc1_avg.reset()
            self.model.acc5_avg.reset()
            self.model.acc10_avg.reset()
            self.model.acc20_avg.reset()
            self.model.mrr_avg.reset()
            for idx, batch in enumerate(self.val_dataloader):
                batch = self.move_batch_to_device(batch, self.gpu)
                with torch.no_grad():
                    with autocast('cuda', enabled=self.use_amp):
                        self.model.validation_step(batch, idx)
            poi_val_loss = self.model.poi_loss_avg.avg
            acc1_val = self.model.acc1_avg.avg
            acc5_val = self.model.acc5_avg.avg
            acc10_val = self.model.acc10_avg.avg
            acc20_val = self.model.acc20_avg.avg
            mrr_val = self.model.mrr_avg.avg
            
            print(f"Epoch {self.epoch}, POI validation loss = {poi_val_loss}, Time taken: {int((time.time() - epoch_start_time)//60)}:{int((time.time() - epoch_start_time)%60)} minutes")
            print(f"Epoch {self.epoch}, POI ACC@1 = {acc1_val}, ACC@5 = {acc5_val}, ACC@10 = {acc10_val}, ACC@20 = {acc20_val}")
            print(f"Epoch {self.epoch}, POI MRR = {mrr_val}")
            
            
            if self.ray_tuner:
                self.ray_tuner.report(
                    {'loss': poi_val_loss.detach().cpu().item(), 'accuracy': acc5_val.detach().cpu().item()},
                    # checkpoint=checkpoint
                )
        