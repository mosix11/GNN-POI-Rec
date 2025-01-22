import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl

from ..metrics import MRR, AccuracyK
from ..dataset import FoursquareNYC


class TrajLSTM(pl.LightningModule):

    def __init__(
        self,
        dataset: FoursquareNYC,
        user_emb_dim: int = 512,
        poi_emb_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        lstm_dropout: float = 0.5,
        emb_dropout: float = 0.5,
        optim_lr: float = 1e-4,
        optim_type: str = "adamw",
    ) -> None:
        super().__init__()

        num_user = dataset.STATS["num_user"] + 1  # index 0 is padding
        num_pois = dataset.STATS["num_pois"] + 1  # index 0 is padding

        self.num_user = num_user
        self.user_emb_dim = user_emb_dim
        self.num_poi = num_pois
        self.poi_emb_dim = poi_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.optim_lr = optim_lr
        self.optim_type = optim_type

        self.user_emb = nn.Embedding(
            num_embeddings=num_user, embedding_dim=user_emb_dim, padding_idx=0
        )
        self.poi_emb = nn.Embedding(
            num_embeddings=num_pois, embedding_dim=poi_emb_dim, padding_idx=0
        )

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.net = nn.LSTM(
            input_size=poi_emb_dim + user_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.out_projector = nn.Linear(in_features=hidden_dim, out_features=num_pois)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.acc1 = AccuracyK(1)
        self.acc5 = AccuracyK(5)
        self.acc10 = AccuracyK(10)
        self.acc20 = AccuracyK(20)
        self.mrr = MRR()

    def forward(self, user_ids, pois, orig_lengths):
        """
        Forward pass logic
        user_ids: Tensor of shape (batch_size, seq_len)
        pois: Tensor of shape (batch_size, seq_len)
        Returns logits of shape (batch_size, seq_len, num_poi)
        """
        user_embs = self.user_emb(user_ids)
        poi_embs = self.poi_emb(pois)

        inputs = torch.cat(
            [poi_embs, user_embs], dim=-1
        )  # shape (batch, seq_len, poi_emb_dim + user_emb_dim)
        inputs = self.emb_dropout(inputs)

        pck_inputs = pack_padded_sequence(
            inputs,
            lengths=orig_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,  # or True if you're sure it's sorted
        )

        pck_output, _ = self.net(pck_inputs)

        lstm_out, _ = pad_packed_sequence(
            pck_output, batch_first=True
        )  # shape (batch, seq_len, hidden_dim)

        # out = torch.cat((lstm_out, self.emb_dropout(user_embs)), dim=2) # shape (bathc, seq_len, hidden_dim + user_emb_dim)
        logits = self.out_projector(lstm_out)  # shape (batch, seq_len, num_poi)

        return logits

    def training_step(self, batch, batch_idx):
        # Training step logic
        """
        Batch contains x, y, original lengths,
        where x and y contain (in order):
            [User ID, POIs, POIs Category, POIs Geohash, POIs Time Slot, POIs Unix Timestamp]
            Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
        - y, is the shifted version of x by 1 time step (teacher forcing)

        """
        x, y, orig_lengths = batch
        user_ids, pois = (
            x[0],
            x[1],
        )  # User ID shape [batch], POIs shape [batch, seq_len]
        seq_len = pois.size(1)

        mask = torch.arange(seq_len, device=user_ids.device).expand(
            len(orig_lengths), seq_len
        ) < orig_lengths.unsqueeze(1)
        # user IDs are the same for all elements in the sequence (trajectory)
        user_ids = user_ids.unsqueeze(1).repeat(1, seq_len)  # shape [batch, seq_len]
        user_ids *= mask

        logits = self.forward(user_ids, pois, orig_lengths)

        logits = logits.view(-1, self.num_poi)
        true_pois = y[1]  # shifted POIs (teacher forcing) shape [batch, seq_len]
        true_pois = true_pois.reshape(-1)

        loss = self.loss_fn(logits, true_pois)

        self.log("Train/Loss", loss, on_epoch=True, reduce_fx="mean", prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step logic
        """
        Batch contains x, y, original lengths,
        where x contians (in order):
            [User ID, POIs, POIs Category, POIs Geohash, POIs Time Slot, POIs Unix Timestamp]
            Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
        - y also contains the same elements but it contains the last sliced ordered check-ins of the user for test
        - elements in y are check-ins that the model has not seen during training
        - the number of elements depends on the `num_test_checkins` of `FoursquareNYC`

        ** The collate function pads the sequences in the batch to maximum length for testing.
        ** In each validation step the model only predicts the first element in the test samples
        ** Full testing for all test samples will be don after training is finished (refer to evaluate method)

        """
        x, y, orig_lengths = batch
        user_ids, pois = (
            x[0],
            x[1],
        )  # User ID shape [batch], POIs shape [batch, seq_len]
        seq_len = pois.size(1)

        mask = torch.arange(seq_len, device=user_ids.device).expand(
            len(orig_lengths), seq_len
        ) < orig_lengths.unsqueeze(1)
        user_ids = user_ids.unsqueeze(1).repeat(1, seq_len)  # shape [batch, seq_len]
        user_ids *= mask

        logits = self.forward(
            user_ids, pois, orig_lengths
        )  # shape (batch, seq_len, num_poi)

        last_valid_indices = orig_lengths - 1
        batch_indices = torch.arange(logits.size(0), device=user_ids.device)
        last_logit = logits[
            batch_indices, last_valid_indices, :
        ]  # shape (batch, num_poi)

        true_pois = y[1][:, 0]  # take the first POI in the test check-ins shape (batch)
        true_pois = true_pois.reshape(-1)

        loss = self.loss_fn(last_logit, true_pois)

        acc1 = self.acc1(last_logit, true_pois)
        acc5 = self.acc5(last_logit, true_pois)
        acc10 = self.acc10(last_logit, true_pois)
        acc20 = self.acc20(last_logit, true_pois)
        mrr = self.mrr(last_logit, true_pois)

        self.log("Val/Loss", loss, on_epoch=True, reduce_fx="mean", prog_bar=True)
        self.log("Val/Acc@1", acc1, on_epoch=True, reduce_fx="mean", prog_bar=True)
        self.log("Val/Acc@5", acc5, on_epoch=True, reduce_fx="mean", prog_bar=True)
        self.log("Val/Acc@10", acc10, on_epoch=True, reduce_fx="mean", prog_bar=True)
        self.log("Val/Acc@20", acc20, on_epoch=True, reduce_fx="mean", prog_bar=True)
        self.log("Val/MRR", mrr, on_epoch=True, reduce_fx="mean", prog_bar=True)

    def test_step(self, batch, batch_idx): ...

    def configure_optimizers(self):
        """
        Optimizer configuration
        """
        if self.optim_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        elif self.optim_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_lr)
        return optimizer
