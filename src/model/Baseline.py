import torch
import torch.nn as nn

import lightning.pytorch as pl

from ..metrics import MRR, AccuracyK, Accuracy1


class TrajLSTM(pl.LightningModule):
    
    def __init__(self,
                 num_user:int = None,
                 user_emb_dim:int = 128,
                 num_poi:int = None,
                 poi_emb_dim:int = 128,
                 hidden_dim:int = 256,
                 num_layers:int = 1,
                 output_dim:int = 128,
                 dropout:int = 0,
                 ):
        super().__init__()
        
        assert num_user != None, 'The number of Users should be specified.'
        assert num_poi != None, 'The number of POIs should be specified.'
        
        self.num_user = num_user
        self.user_emb_dim = user_emb_dim
        self.num_poi = num_poi
        self.poi_emb_dim = poi_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.user_emb = nn.Embedding(num_embeddings=num_user,
                                     embedding_dim=user_emb_dim,
                                     padding_idx=0)
        self.poi_emb = nn.Embedding(num_embeddings=num_poi,
                                    embedding_dim=poi_emb_dim,
                                    padding_idx=0)
        self.net = nn.LSTM(input_size=poi_emb_dim+user_emb_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)
        
        
        
        
    def forward(self, x):
        # Forward pass logic
        pass

    def training_step(self, batch):
        # Training step logic
        pass

    def validation_step(self, batch):
        # Validation step logic
        pass

    def configure_optimizers(self):
        # Optimizer configuration
        pass