import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl

from tabulate import tabulate

from ..metrics import MRR, AccuracyK
from ..dataset import FoursquareNYC

from ..utils import misc_utils


class TrajLSTM(pl.LightningModule):
    """
    Baseline model for processing trajectories of user check-ins and
    producing a probablity over next POI using a simple LSTM.
    """

    def __init__(
        self,
        dataset: FoursquareNYC,
        user_emb_dim: int = 512,
        poi_emb_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,
        emb_dropout: float = 0.5,
        optim_lr: float = 1e-4,
        optim_type: str = "adamw",
    ) -> None:
        super().__init__()
        """
        The init method is responsible for initializing model layers and properties.
        
        Args:
            `user_emb_dim`: (int): The embedding dimension of the User IDs.
            `poi_emb_dim`: (int): The embedding dimension of the POI IDs.
            `hidden_dim`: (int): The dimension of the hidden units of LSTM
            `num_layers` (int): Number of LSTM layers.
            `emb_dropout`: (float): Dropout rate for all embeddings.
            `optim_lr`: (float): Learning rate for the optimizer.
            `optim_type`: (str): Type of the optimizer. Between [`adam`, `adamw`].
        """

        num_user = dataset.STATS["num_user"] + 1  # index 0 is padding
        num_pois = dataset.STATS["num_pois"] + 1  # index 0 is padding

        self.num_user = num_user
        self.user_emb_dim = user_emb_dim
        self.num_pois = num_pois
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

        self.poi_loss_avg = misc_utils.AverageMeter()
        self.acc1_avg = misc_utils.AverageMeter()
        self.acc5_avg = misc_utils.AverageMeter()
        self.acc10_avg = misc_utils.AverageMeter()
        self.acc20_avg = misc_utils.AverageMeter()
        self.mrr_avg = misc_utils.AverageMeter()

    def forward(self, user_ids, pois, orig_lengths):
        """
        Forward pass logic
        `user_ids`: Tensor of shape (batch, seq_len)
        `pois`: Tensor of shape (batch, seq_len)
        `orig_lengths`: Tensor of shape (batch)
        Returns logits of shape (batch, seq_len, num_poi)
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
            enforce_sorted=False,  # Since the sequences are not sorted
        )

        pck_output, _ = self.net(pck_inputs)

        lstm_out, _ = pad_packed_sequence(
            pck_output, batch_first=True
        )  # shape (batch, seq_len, hidden_dim)

        # out = torch.cat((lstm_out, self.emb_dropout(user_embs)), dim=2) # shape (bathc, seq_len, hidden_dim + user_emb_dim)
        logits = self.out_projector(lstm_out)  # shape (batch, seq_len, num_poi)

        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step logic.

        Batch contains x, y, original lengths,
        where x and y contain (in order):
            [User ID, POIs, POIs Category, POIs Geohashes, POIs Time Slot, POIs Unix Timestamp]
            Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
        - y, is the shifted version of x by 1 time step (teacher forcing)

        Args:
            batch: (List[Tensor]): The training batch.
        Returns:
            training loss.
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

        logits = logits.view(-1, self.num_pois)
        true_pois = y[1]  # shifted POIs (teacher forcing) shape [batch, seq_len]
        true_pois = true_pois.reshape(-1)

        loss = self.loss_fn(logits, true_pois)

        self.poi_loss_avg.update(loss.detach().cpu().item())

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        self.poi_loss_avg.reset()
        self.logger.experiment.add_scalar(
            "Learning Rate",
            self.optimizers(use_pl_optimizer=False).param_groups[0]["lr"],
            self.current_epoch,
        )
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_epoch_end(self):
        self.logger.experiment.add_scalar(
            "Train/POI Loss", self.poi_loss_avg.avg, self.current_epoch
        )
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        """
        Validation step logic.

        Batch contains x, y, original lengths,
        where x contians (in order):
            [User ID, POIs, POIs Category, POIs Geohash, POIs Time Slot, POIs Unix Timestamp]
            Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
        - y also contains the same elements but it contains the last sliced ordered check-ins of the user
        - the number of elements depends on the `num_test_checkins` of `FoursquareNYC`

        ** The collate function pads the sequences in the batch to maximum length for testing.
        ** In each validation step the model only predicts the first element in the test samples
        ** Full testing for all test samples will be don after training is finished (refer to the test_step method)

        Args:
            batch: (List[Tensor]): The validation batch.
        Returns:
            validation loss.
        """
        x, y, orig_lengths = batch
        user_ids, pois = (
            x[0],
            x[1],
        )  # User ID shape [batch], POIs shape [batch, seq_len]
        tgt_pois = y[1]

        batch_size = pois.shape[0]
        seq_len = pois.size(1)
        target_seq_len = tgt_pois.size(1)

        mask = torch.arange(seq_len, device=user_ids.device).expand(
            len(orig_lengths), seq_len
        ) < orig_lengths.unsqueeze(1)
        user_ids = user_ids.unsqueeze(1).repeat(1, seq_len)  # shape [batch, seq_len]
        user_ids *= mask

        poi_logits = self.forward(
            user_ids, pois, orig_lengths
        )  # shape (batch, seq_len, num_poi)

        last_train_indices = orig_lengths - target_seq_len

        # Here we extract the logits of the last check-in in traning trajectory
        # This logit is used to predict the first check-in in the test check-ins.
        tgt_poi_logits = torch.stack(
            [
                poi_logits[i, last_train_indices[i] : last_train_indices[i] + 1, :]
                for i in range(batch_size)
            ]
        )  # shape (batch, 1, num_poi)

        tgt_poi_logits = tgt_poi_logits.view(-1, self.num_pois)
        # The target is the first check-in in the test check-ins.
        tgt_pois = tgt_pois[:, 0].reshape(-1)  # shape (batch, 1, num_poi)

        poi_loss = self.loss_fn(tgt_poi_logits, tgt_pois.reshape(-1))

        acc1_val = self.acc1(tgt_poi_logits, tgt_pois)
        acc5_val = self.acc5(tgt_poi_logits, tgt_pois)
        acc10_val = self.acc10(tgt_poi_logits, tgt_pois)
        acc20_val = self.acc20(tgt_poi_logits, tgt_pois)
        mrr_val = self.mrr(tgt_poi_logits, tgt_pois)

        self.poi_loss_avg.update(poi_loss)
        self.acc1_avg.update(acc1_val.detach().cpu().item())
        self.acc5_avg.update(acc5_val.detach().cpu().item())
        self.acc10_avg.update(acc10_val.detach().cpu().item())
        self.acc20_avg.update(acc20_val.detach().cpu().item())
        self.mrr_avg.update(mrr_val.detach().cpu().item())

        self.log("Val/Loss", poi_loss, prog_bar=False, on_epoch=True, logger=True)
        return poi_loss

    def on_validation_epoch_start(self):
        self.poi_loss_avg.reset()
        self.acc1_avg.reset()
        self.acc5_avg.reset()
        self.acc10_avg.reset()
        self.acc20_avg.reset()
        self.mrr_avg.reset()

    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalar(
            "Val/POI Loss", self.poi_loss_avg.avg, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Val/Acc@1", self.acc1_avg.avg, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Val/Acc@5", self.acc5_avg.avg, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Val/Acc@10", self.acc10_avg.avg, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Val/Acc@20", self.acc20_avg.avg, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "Val/MRR", self.mrr_avg.avg, self.current_epoch
        )

        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        """
        Test phase logic.

        Batch contains x, y, original lengths,
        where x contians (in order):
            [User ID, POIs, POIs Category, POIs Geohash, POIs Time Slot, POIs Unix Timestamp]
            Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
        - y also contains the same elements but it contains the last sliced ordered check-ins of the user
        - Tthe number of test check-ins depends on the `num_test_checkins` of `FoursquareNYC`

        ** The collate function pads the sequences in the batch to maximum sequence length for testing.
        ** In test phase we input the entire user trajectory to the model and slice the logits corresponding
        ** to the test check-ins.

        """
        x, y, orig_lengths = batch
        user_ids, pois = (
            x[0],
            x[1],
        )  # User ID shape [batch], POIs shape [batch, seq_len]
        tgt_pois = y[1]

        batch_size = pois.shape[0]
        seq_len = pois.size(1)
        target_seq_len = tgt_pois.size(1)

        mask = torch.arange(seq_len, device=user_ids.device).expand(
            len(orig_lengths), seq_len
        ) < orig_lengths.unsqueeze(1)
        user_ids = user_ids.unsqueeze(1).repeat(1, seq_len)  # shape [batch, seq_len]
        user_ids *= mask

        poi_logits = self.forward(
            user_ids, pois, orig_lengths
        )  # shape (batch, seq_len, num_poi)

        # For prediction we only use the logits related to the test visits which start
        # from the logits produced for the last training sample at index `last_train_indices`
        # to `orig_lengths - 1` or simply `last_train_indices:orig_lengths`
        last_train_indices = orig_lengths - target_seq_len

        tgt_poi_logits = torch.stack(
            [
                poi_logits[i, last_train_indices[i] : orig_lengths[i], :]
                for i in range(batch_size)
            ]
        )  # shape (batch, target_seq_len, num_poi)

        tgt_poi_logits = tgt_poi_logits.view(-1, self.num_pois)

        tgt_pois = tgt_pois.reshape(-1)

        poi_loss = self.loss_fn(tgt_poi_logits, tgt_pois.reshape(-1))

        acc1_val = self.acc1(tgt_poi_logits, tgt_pois)
        acc5_val = self.acc5(tgt_poi_logits, tgt_pois)
        acc10_val = self.acc10(tgt_poi_logits, tgt_pois)
        acc20_val = self.acc20(tgt_poi_logits, tgt_pois)
        mrr_val = self.mrr(tgt_poi_logits, tgt_pois)

        self.poi_loss_avg.update(poi_loss)
        self.acc1_avg.update(acc1_val.detach().cpu().item())
        self.acc5_avg.update(acc5_val.detach().cpu().item())
        self.acc10_avg.update(acc10_val.detach().cpu().item())
        self.acc20_avg.update(acc20_val.detach().cpu().item())
        self.mrr_avg.update(mrr_val.detach().cpu().item())

    def on_test_epoch_start(self):
        self.poi_loss_avg.reset()
        self.acc1_avg.reset()
        self.acc5_avg.reset()
        self.acc10_avg.reset()
        self.acc20_avg.reset()
        self.mrr_avg.reset()
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        # Report the perfromance of the model.
        headers = ["Metric", "Score"]
        table = [
            ["Acc@1", self.acc1_avg.avg],
            ["Acc@5", self.acc5_avg.avg],
            ["Acc@10", self.acc10_avg.avg],
            ["Acc@20", self.acc20_avg.avg],
            ["MRR", self.mrr_avg.avg],
        ]
        print("\n\n")
        print(tabulate(table, headers, tablefmt="rounded_grid"))
        print("\n\n")
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        """
        Optimizer configuration.
        Based on `optimizer_type` hyperparameter, one of Adam or AdamW optimizers
        will be initialized.
        For learning rate scheduling, ReduceLROnPlateau scheduler is used.
        """
        if self.optim_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        elif self.optim_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, mode="min", patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "Val/Loss",
        }
