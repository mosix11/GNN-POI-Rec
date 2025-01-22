import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl

from ..metrics import MRR, AccuracyK
from ..dataset import FoursquareNYC


class HTM_GRN(pl.LightningModule):

    def __init__(
        self,
        dataset: FoursquareNYC,
        user_emb_dim: int = 512,
        poi_emb_dim: int = 512,
        poi_cat_emb_dim: int = 256,
        gh_emb_dim: int = 512,
        hidden_dim: int = 1024,
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0.5,
        emb_dropout: float = 0.5,
        num_GAT_heads: int = 4,
        GAT_dropout: float = 0.0,
        optim_lr: float = 1e-4,
        optim_type: str = "adamw",
    ) -> None:

        super().__init__()

        self.hierarchical_spatial_graph = dataset.hierarchical_spatial_graph
        self.temporal_graph = dataset.temporal_graph
        self.spatial_graph = dataset.spatial_graph
        self.poi_trajectories = dataset.poi_trajectories

        self.num_users = dataset.STATS["num_user"] + 1  # index 0 is padding
        self.num_pois = dataset.STATS["num_pois"] + 1  # index 0 is padding
        self.num_poi_cat = dataset.STATS["num_poi_cat"] + 1  # index 0 is padding
        self.geohash_precision = dataset.geohash_precision
        self.num_ghs = [
            dataset.STATS[f"num_gh_P{precision}"] + 1  # index 0 is padding
            for precision in self.geohash_precision
        ]

        self.user_emb_dim = user_emb_dim
        self.poi_emb_dim = poi_emb_dim
        self.poi_cat_emb_dim = poi_cat_emb_dim
        self.gh_emb_dim = gh_emb_dim
        self.hidden_dim = hidden_dim

        self.user_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=user_emb_dim, padding_idx=0
        )
        self.poi_emb = nn.Embedding(
            num_embeddings=self.num_pois, embedding_dim=poi_emb_dim, padding_idx=0
        )
        self.poi_cat_emb = nn.Embedding(
            num_embeddings=self.num_poi_cat,
            embedding_dim=poi_cat_emb_dim,
            padding_idx=0,
        )
        self.gh_embeddings = [
            nn.Embedding(num_embeddings=num_gh, embedding_dim=gh_emb_dim, padding_idx=0)
            for num_gh in self.num_ghs
        ]

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.temporalGAT = GATConv(
            poi_emb_dim,
            poi_emb_dim,
            heads=num_GAT_heads,
            concat=True,
            dropout=GAT_dropout,
        )
        self.spatialGAT = GATConv(
            poi_emb_dim,
            poi_emb_dim,
            heads=num_GAT_heads,
            concat=True,
            dropout=GAT_dropout,
        )

        self.net = nn.LSTM(
            input_size=2 * poi_emb_dim + user_emb_dim + poi_cat_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # This single loss function is applicable to all tasks since the padding index for all
        # embeddings are 0.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.acc1 = AccuracyK(1)
        self.acc5 = AccuracyK(5)
        self.acc10 = AccuracyK(10)
        self.acc20 = AccuracyK(20)
        self.mrr = MRR()
