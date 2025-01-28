import torch
import torch.nn as nn
import numpy as np
import torch_geometric.nn as tgnn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl
from tabulate import tabulate

from ..metrics import MRR, AccuracyK
from ..dataset import FoursquareNYC
from ..utils import misc_utils

from typing import List


class HMT_GRN_V3(pl.LightningModule):

    def __init__(
        self,
        dataset: FoursquareNYC,
        user_emb_dim: int = 256,
        poi_emb_dim: int = 512,
        poi_cat_emb_dim: int = 128,
        gh_emb_dim: int = 128,
        ts_emb_dim: int = 128,
        hidden_dim: int = 512,
        emb_switch: List[bool] = [
            True,
            True,
            False,
            True,
        ],  # [user_emb, poi_emb, poi_cat_emb, ts_emb]
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0.,
        emb_dropout: float = 0.9,
        num_GAT_heads: int = 4,
        GAT_dropout: float = 0.5,
        # task_loss_coefficients: List[float] = [1., 1., 1., 1.],
        task_loss_coefficients: List[float] = [1., 0., 0.5, 0.],
        optim_lr: float = 0.0001,
        optim_type: str = "adamw",
    ) -> None:

        super().__init__()

        assert (
            emb_switch[0] and emb_switch[1]
        ), "User embeddings and POI embeddings must be used and can't be turned off!"
        
        
        self.hierarchical_spatial_graph = dataset.hierarchical_spatial_graph
        self.register_buffer("spatial_graph", dataset.spatial_graph)
        self.register_buffer("spatial_graph_edge_COO", dataset.get_spatial_graph_edge_indices(self_loop=True)[0])
        self.register_buffer("temporal_graph", dataset.temporal_graph)
        self.register_buffer("temporal_graph_edge_COO", dataset.get_temporal_graph_edge_indices(self_loop=True)[0])
        
        
        self.poi_trajectories = dataset.poi_trajectories

        self.num_users = dataset.STATS["num_user"] + 1  # index 0 is padding
        self.num_pois = dataset.STATS["num_pois"] + 1  # index 0 is padding
        self.num_poi_cat = dataset.STATS["num_poi_cat"] + 1  # index 0 is padding
        self.num_ts = dataset.STATS["num_time_slots"] + 1  # index 0 is padding
        self.geohash_precision = dataset.geohash_precision
        self.num_ghs = [
            dataset.STATS[f"num_gh_P{precision}"] + 1  # index 0 is padding
            for precision in self.geohash_precision
        ]
        
        assert (
            len(task_loss_coefficients) - 1 == len(self.geohash_precision)
        ), "You should provide one coefficient for next POI prediction task and one for each next geohash prediction task."
        
        self.task_loss_coefficients = task_loss_coefficients

        self.user_emb_dim = user_emb_dim
        self.poi_emb_dim = poi_emb_dim
        self.poi_cat_emb_dim = poi_cat_emb_dim if emb_switch[2] else 0
        self.gh_emb_dim = gh_emb_dim
        self.ts_emb_dim = ts_emb_dim if emb_switch[3] else 0
        self.hidden_dim = hidden_dim

        self.emb_switch = emb_switch

        self.aggregated_emb_dim = np.sum(
            [self.poi_emb_dim, self.poi_cat_emb_dim, self.ts_emb_dim]
        )

        self.optim_lr = optim_lr
        self.optim_type = optim_type

        self.user_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=user_emb_dim, padding_idx=0
        )
        self.poi_emb = nn.Embedding(
            num_embeddings=self.num_pois, embedding_dim=poi_emb_dim, padding_idx=0
        )
        self.poi_cat_emb = (
            nn.Embedding(
                num_embeddings=self.num_poi_cat,
                embedding_dim=poi_cat_emb_dim,
                padding_idx=0,
            )
            if emb_switch[2]
            else None
        )

        self.ts_emb = (
            nn.Embedding(
                num_embeddings=self.num_ts, embedding_dim=ts_emb_dim, padding_idx=0
            )
            if emb_switch[3]
            else None
        )

        self.gh_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_gh, embedding_dim=gh_emb_dim, padding_idx=0)
            for num_gh in self.num_ghs
        ])

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.temporal_GAT = tgnn.GATConv(
            poi_emb_dim,
            poi_emb_dim,
            heads=num_GAT_heads,
            concat=False,
            dropout=GAT_dropout,
            add_self_loops=False
        )
        self.spatial_GAT = tgnn.GATConv(
            poi_emb_dim,
            poi_emb_dim,
            heads=num_GAT_heads,
            concat=False,
            dropout=GAT_dropout,
            add_self_loops=False
        )
        
        self.attended_poi_projector = nn.Linear(
            in_features=3*poi_emb_dim, out_features=poi_emb_dim
        )

        self.net = nn.LSTM(
            input_size=self.aggregated_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.poi_final_projector = nn.Linear(
            in_features=self.user_emb_dim + hidden_dim, out_features=self.num_pois
        )
        self.geohash_projectors = nn.ModuleList([
            nn.Linear(in_features=hidden_dim + gh_emb_dim, out_features=num_gh)
            for num_gh in self.num_ghs
        ])
        

        # This single loss function is applicable to all tasks since the padding index for all
        # embeddings are 0.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        self.acc1 = AccuracyK(1)
        self.acc5 = AccuracyK(5)
        self.acc10 = AccuracyK(10)
        self.acc20 = AccuracyK(20)
        self.mrr = MRR()
        
        self.poi_loss_avg = misc_utils.AverageMeter()
        self.geohash_losses_avg = [
            misc_utils.AverageMeter() for idx in range(len(self.geohash_precision))
        ]
        self.acc1_avg = misc_utils.AverageMeter()
        self.acc5_avg = misc_utils.AverageMeter()
        self.acc10_avg = misc_utils.AverageMeter()
        self.acc20_avg = misc_utils.AverageMeter()
        self.mrr_avg = misc_utils.AverageMeter()
        

    def forward(
        self,
        users: torch.Tensor,
        pois: torch.Tensor,
        pois_cat: torch.Tensor,
        ghs: List[torch.Tensor],
        ts: torch.Tensor,
        ut: torch.Tensor,
        orig_lengths: torch.Tensor,
        len_mask: torch.Tensor,
    ):
        """
        Forward pass logic
        Inputs:
            `user_ids`: Tensor of shape (batch, seq_len),
                containing user IDs
            `pois`: Tensor of shape (batch, seq_len),
                containing POI IDs in the trajectory.
            `pois_cat`: Tensor of shape (batch, seq_len),
                containing POI Category IDs.
            `ghs`: List of Tensors of shape (num_gehash_precisions, batch, seq_len),
                containig geohash IDs at different precision levels.
            `ts`: Tensor of shape (batch, seq_len),
                containig weekly time slot IDs from 1 to 57 (0 is padding).
            `ut`: Tensor of shape (batch, seq_len),
                containign unix timestamp for each visit.
            `orig_lengths`: Tensor of shape (batch),
                containig the original length of each trajectory (sequences are padded).
            `len_mask`: Tensor of shape (batch, seq_len),
                containing the mask for lengths.

        Returns:
            logits of shape (batch, seq_len, num_poi)
        """
        batch_size = users.shape[0]
        seq_len = users.shape[1]

        user_embs = self.user_emb(users)

        poi_cat_embs = self.poi_cat_emb(pois_cat) if self.emb_switch[2] else None
        ts_embs = self.ts_emb(ts) if self.emb_switch[3] else None
        gh_embs = [
            # self.gh_embeddings[idx](ghs[idx]) if ghs[idx] is not None else None
            self.gh_embeddings[idx](ghs[idx])
            for idx in range(len(self.geohash_precision))
        ]

        # Apply the Graph Attention operation on all nodes in the graph.
        # shape of all three matrices below: (num_pois, poi_emb_dim)
        all_pois_embs = self.poi_emb(torch.arange(self.num_pois, device=pois.device))
        spatially_attended_poi_embs = self.spatial_GAT(all_pois_embs, self.spatial_graph_edge_COO) 
        temporally_attended_poi_embs = self.temporal_GAT(all_pois_embs, self.temporal_graph_edge_COO)
        
        # Extract the embeddings of the nodes present in the `pois` tensor
        # shape of all three tensors below: (batch, seq_len, poi_emb_dim)
        selected_all_pois_embs = all_pois_embs[pois]  
        selected_spatially_attended_poi_embs = spatially_attended_poi_embs[pois]  
        selected_temporally_attended_poi_embs = temporally_attended_poi_embs[pois]  


        
        final_poi_emb = self.attended_poi_projector(
            torch.cat(
                [selected_all_pois_embs,
                 selected_spatially_attended_poi_embs,
                 selected_temporally_attended_poi_embs],
                dim=-1
                )
        )
                    
        lstm_inputs = final_poi_emb # shape (batch, seq_len, poi_emb_dim)
        
        if poi_cat_embs is not None:
            # shape (batch, seq_len, poi_emb_dim  + poi_cat_emb_dim)
            lstm_inputs = torch.cat([lstm_inputs, poi_cat_embs], dim=-1) 
        if ts_embs is not None:
            # shape (batch, seq_len, poi_emb_dim  + poi_cat_emb_dim + ts_emb_dim)
            lstm_inputs = torch.cat([lstm_inputs, ts_embs], dim=-1) 

        lstm_inputs = self.emb_dropout(lstm_inputs)
                
            

        pck_inputs = pack_padded_sequence(
            lstm_inputs,
            lengths=orig_lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,  # Since the sequences are not sorted
        )

        pck_output, _ = self.net(pck_inputs)

        lstm_out, _ = pad_packed_sequence(
            pck_output, batch_first=True
        )  # shape (batch, seq_len, hidden_dim)

        poi_lstm_out = torch.cat([lstm_out, user_embs], dim=-1)
        poi_logits = self.poi_final_projector(poi_lstm_out)
        
        gh_embs = [
            self.emb_dropout(gh_emb) for gh_emb in gh_embs
        ]
        
        ghs_logits = [
            self.geohash_projectors[idx](torch.cat([lstm_out, gh_embs[idx]], dim=-1))
            for idx in range(len(self.geohash_precision)) 
        ]
        

        return poi_logits, ghs_logits


    def training_step(self, batch, batch_idx):
        # Training step logic
        """
        Batch contains x, y, original lengths,
        where x and y contain (in order):
            [User ID, POIs, POIs Category, POIs Geohashes, POIs Time Slot, POIs Unix Timestamp]
            Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
        - y, is the shifted version of x by 1 time step (teacher forcing)

        """
        x, y, orig_lengths = batch

        # User ID shape [batch]
        # POIs, POIs Cat, Geohash Vectors, Time Slots, and Unix Timestamp shape [batch, seq_len]
        # Target values are a shifted version of the `x` by one time step (teacher forcing)
        if len(self.geohash_precision) == 1:
            users, pois, pois_cat, gh1, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1 = y
            gh2, gh3, tgt_gh2, tgt_gh3 = None, None, None, None
        elif len(self.geohash_precision) == 2:
            users, pois, pois_cat, gh1, gh2, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2 = y
            gh3, tgt_gh3 = None
        elif len(self.geohash_precision) == 3:
            users, pois, pois_cat, gh1, gh2, gh3, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2, tgt_gh3 = y
        else:
            raise RuntimeError(
                "The case for more than 3 geohash precisions in not handeled."
            )

        batch_size = pois.size(0)
        seq_len = pois.size(1)
        device = users.device
        

        # This mask contains 1 in places that the sequences does not contain pad token
        mask = torch.arange(seq_len, device=device).expand(
            len(orig_lengths), seq_len
        ) < orig_lengths.unsqueeze(1)

        # user IDs are the same for all elements in the sequence (trajectory)
        users = users.unsqueeze(1).repeat(1, seq_len)  # shape [batch, seq_len]
        users *= mask

        poi_logits, ghs_logits = self.forward(
            users, pois, pois_cat, [gh1, gh2, gh3], ts, ut, orig_lengths, mask
        )

        poi_logits = poi_logits.view(-1, self.num_pois)
        ghs_logits = [
            ghs_logits[idx].view(-1, self.num_ghs[idx])
            for idx in range(len(self.geohash_precision))
        ]

        poi_loss = self.loss_fn(poi_logits, tgt_pois.reshape(-1))
        
        tgt_ghs = [tgt_gh1, tgt_gh2, tgt_gh3]
        gh_losses = [
            self.loss_fn(ghs_logits[idx], tgt_ghs[idx].reshape(-1))
            for idx in range(len(self.geohash_precision))
        ]
        
        poi_loss_weighted = poi_loss * self.task_loss_coefficients[0]
        gh_losses_weighted = [
            gh_losses[idx] * self.task_loss_coefficients[idx + 1]
            for idx in range(len(self.geohash_precision))
        ]
        
        total_loss = poi_loss_weighted + sum(gh_losses_weighted)
        
        self.poi_loss_avg.update(poi_loss.detach().cpu().item())
        for idx, gh_loss in enumerate(gh_losses):
            self.geohash_losses_avg[idx].update(gh_loss.detach().cpu().item())
        
        return total_loss / 4

    def on_train_batch_start(self, batch, batch_idx):
        self.poi_loss_avg.reset()
        for idx in range(len(self.geohash_precision)):
            self.geohash_losses_avg[idx].reset()
        self.logger.experiment.add_scalar(
            "Learning Rate",
            self.optimizers(use_pl_optimizer=False).param_groups[0]["lr"],
            self.current_epoch,
        )
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_epoch_end(self):
        self.logger.experiment.add_scalar("Train/POI Loss", self.poi_loss_avg.avg, self.current_epoch)
        for idx in range(len(self.geohash_precision)):
            self.logger.experiment.add_scalar(f"Train/Geohash P{self.geohash_precision[idx]} Loss", self.geohash_losses_avg[idx].avg, self.current_epoch)
        return super().on_train_epoch_end()
    
    
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

        # User ID shape [batch]
        # POIs, POIs Cat, Geohash Vectors, Time Slots, and Unix Timestamp shape [batch, seq_len]
        # Target values are unseen check-ins from user trajectories.
        if len(self.geohash_precision) == 1:
            users, pois, pois_cat, gh1, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1 = y
            gh2, gh3, tgt_gh2, tgt_gh3 = None, None, None, None
        elif len(self.geohash_precision) == 2:
            users, pois, pois_cat, gh1, gh2, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2 = y
            gh3, tgt_gh3 = None
        elif len(self.geohash_precision) == 3:
            users, pois, pois_cat, gh1, gh2, gh3, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2, tgt_gh3 = y
        else:
            raise RuntimeError(
                "The case for more than 3 geohash precisions in not handeled."
            )

        batch_size = pois.size(0)
        seq_len = pois.size(1)
        target_seq_len = tgt_pois.size(1)
        device = users.device
        
        
        # This mask contains 1 in places that the sequences does not contain pad token
        mask = torch.arange(seq_len, device=device).expand(
            len(orig_lengths), seq_len
        ) < orig_lengths.unsqueeze(1)

        # user IDs are the same for all elements in the sequence (trajectory)
        users = users.unsqueeze(1).repeat(1, seq_len)  # shape [batch, seq_len]
        users *= mask

        poi_logits, ghs_logits = self.forward(
            users, pois, pois_cat, [gh1, gh2, gh3], ts, ut, orig_lengths, mask
        )
        

        # For prediction we only use the logits related to the test visits which start 
        # from the logits produced for the last training sample at index `last_train_indices`
        # to `orig_lengths - 1` or simply `last_train_indices:orig_lengths`

        last_train_indices = orig_lengths - target_seq_len
        
        
        tgt_poi_logits = torch.stack(
            [poi_logits[i, last_train_indices[i]:orig_lengths[i], :] for i in range(batch_size)]
            ) # shape (batch, target_seq_len, num_poi)
        
        tgt_ghs_logits = [
            torch.stack(
                [
                    ghs_logits[idx][i, last_train_indices[i]:orig_lengths[i], :]
                    for i in range(batch_size)
                ]
            ) for idx in range(len(self.geohash_precision))
        ] # shape (num_geohash_precisions, (batch, target_seq_len, num_gh@P))
        
        
        tgt_poi_logits = tgt_poi_logits.view(-1, self.num_pois)
        tgt_ghs_logits = [
            tgt_ghs_logits[idx].view(-1, self.num_ghs[idx])
            for idx in range(len(self.geohash_precision))
        ]
        
        tgt_pois = tgt_pois.reshape(-1)
        
        tgt_ghs = [tgt_gh1, tgt_gh2, tgt_gh3]

        poi_loss = self.loss_fn(tgt_poi_logits, tgt_pois.reshape(-1))
        gh_losses = [
            self.loss_fn(tgt_ghs_logits[idx], tgt_ghs[idx].reshape(-1))
            for idx in range(len(self.geohash_precision))
        ]
        
        poi_loss_weighted = poi_loss * self.task_loss_coefficients[0]
        gh_losses_weighted = [
            gh_losses[idx] * self.task_loss_coefficients[idx + 1]
            for idx in range(len(self.geohash_precision))
        ]
        
        
        total_loss = poi_loss_weighted + sum(gh_losses_weighted)
        
        acc1_val = self.acc1(tgt_poi_logits, tgt_pois)
        acc5_val = self.acc5(tgt_poi_logits, tgt_pois)
        acc10_val = self.acc10(tgt_poi_logits, tgt_pois)
        acc20_val = self.acc20(tgt_poi_logits, tgt_pois)
        mrr_val = self.mrr(tgt_poi_logits, tgt_pois)
        
        self.poi_loss_avg.update(poi_loss)
        for idx, gh_loss in enumerate(gh_losses):
            self.geohash_losses_avg[idx].update(gh_loss)
        self.acc1_avg.update(acc1_val.detach().cpu().item())
        self.acc5_avg.update(acc5_val.detach().cpu().item())
        self.acc10_avg.update(acc10_val.detach().cpu().item())
        self.acc20_avg.update(acc20_val.detach().cpu().item())
        self.mrr_avg.update(mrr_val.detach().cpu().item())

        self.log("Val/Loss", poi_loss, prog_bar=False, on_epoch=True, logger=True)
        
        return total_loss / 4

        
    def on_validation_epoch_start(self):
        self.poi_loss_avg.reset()
        for idx in range(len(self.geohash_precision)):
            self.geohash_losses_avg[idx].reset()
        self.acc1_avg.reset()
        self.acc5_avg.reset()
        self.acc10_avg.reset()
        self.acc20_avg.reset()
        self.mrr_avg.reset()
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        self.logger.experiment.add_scalar("Val/POI Loss", self.poi_loss_avg.avg, self.current_epoch)
        for idx in range(len(self.geohash_precision)):
            self.logger.experiment.add_scalar(f"Val/Geohash P{self.geohash_precision[idx]} Loss", self.geohash_losses_avg[idx].avg, self.current_epoch)
        self.logger.experiment.add_scalar("Val/Acc@1", self.acc1_avg.avg, self.current_epoch)
        self.logger.experiment.add_scalar("Val/Acc@5", self.acc5_avg.avg, self.current_epoch)
        self.logger.experiment.add_scalar("Val/Acc@10", self.acc10_avg.avg, self.current_epoch)
        self.logger.experiment.add_scalar("Val/Acc@20", self.acc20_avg.avg, self.current_epoch)
        self.logger.experiment.add_scalar("Val/MRR", self.mrr_avg.avg, self.current_epoch)  
        
        return super().on_validation_epoch_end()
    

    def test_step(self, batch, batch_idx):
        """
        Batch contains x, y, original lengths,
        where x contians (in order):
            [User ID, POIs, POIs Category, POIs Geohash, POIs Time Slot, POIs Unix Timestamp]
            Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
        - y also contains the same elements but it contains the last sliced ordered check-ins of the user for test
        - elements in y are check-ins that the model has not seen during training
        - the number of elements depends on the `num_test_checkins` of `FoursquareNYC`

        ** The collate function pads the sequences in the batch to maximum length for testing.
        ** In test phase we input the entire user trajectory to the model and slice the logits corresponding
        ** to the test check-ins.

        """
        x, y, orig_lengths = batch

        # User ID shape [batch]
        # POIs, POIs Cat, Geohash Vectors, Time Slots, and Unix Timestamp shape [batch, seq_len]
        # Target values are unseen check-ins from user trajectories.
        if len(self.geohash_precision) == 1:
            users, pois, pois_cat, gh1, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1 = y
            gh2, gh3, tgt_gh2, tgt_gh3 = None, None, None, None
        elif len(self.geohash_precision) == 2:
            users, pois, pois_cat, gh1, gh2, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2 = y
            gh3, tgt_gh3 = None
        elif len(self.geohash_precision) == 3:
            users, pois, pois_cat, gh1, gh2, gh3, ts, ut = x
            _, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2, tgt_gh3 = y
        else:
            raise RuntimeError(
                "The case for more than 3 geohash precisions in not handeled."
            )

        batch_size = pois.size(0)
        seq_len = pois.size(1)
        target_seq_len = tgt_pois.size(1)
        device = users.device
        
        
        # This mask contains 1 in places that the sequences does not contain pad token
        mask = torch.arange(seq_len, device=device).expand(
            len(orig_lengths), seq_len
        ) < orig_lengths.unsqueeze(1)

        # user IDs are the same for all elements in the sequence (trajectory)
        users = users.unsqueeze(1).repeat(1, seq_len)  # shape [batch, seq_len]
        users *= mask

        poi_logits, ghs_logits = self.forward(
            users, pois, pois_cat, [gh1, gh2, gh3], ts, ut, orig_lengths, mask
        )
        

        # For prediction we only use the logits related to the test visits which start 
        # from the logits produced for the last training sample at index `last_train_indices`
        # to `orig_lengths - 1` or simply `last_train_indices:orig_lengths`
        last_train_indices = orig_lengths - target_seq_len
        
        
        tgt_poi_logits = torch.stack(
            [poi_logits[i, last_train_indices[i]:orig_lengths[i], :] for i in range(batch_size)]
            ) # shape (batch, target_seq_len, num_poi)
        
        tgt_ghs_logits = [
            torch.stack(
                [
                    ghs_logits[idx][i, last_train_indices[i]:orig_lengths[i], :]
                    for i in range(batch_size)
                ]
            ) for idx in range(len(self.geohash_precision))
        ] # shape (num_geohash_precisions, (batch, target_seq_len, num_gh@P))
        
        
        tgt_poi_logits = tgt_poi_logits.view(-1, self.num_pois)
        tgt_ghs_logits = [
            tgt_ghs_logits[idx].view(-1, self.num_ghs[idx])
            for idx in range(len(self.geohash_precision))
        ]
        
        tgt_pois = tgt_pois.reshape(-1)
        
        tgt_ghs = [tgt_gh1, tgt_gh2, tgt_gh3]

        poi_loss = self.loss_fn(tgt_poi_logits, tgt_pois.reshape(-1))
        gh_losses = [
            self.loss_fn(tgt_ghs_logits[idx], tgt_ghs[idx].reshape(-1))
            for idx in range(len(self.geohash_precision))
        ]
        
        poi_loss_weighted = poi_loss * self.task_loss_coefficients[0]
        gh_losses_weighted = [
            gh_losses[idx] * self.task_loss_coefficients[idx + 1]
            for idx in range(len(self.geohash_precision))
        ]
        
        
        total_loss = poi_loss_weighted + sum(gh_losses_weighted)
        
        acc1_val = self.acc1(tgt_poi_logits, tgt_pois)
        acc5_val = self.acc5(tgt_poi_logits, tgt_pois)
        acc10_val = self.acc10(tgt_poi_logits, tgt_pois)
        acc20_val = self.acc20(tgt_poi_logits, tgt_pois)
        mrr_val = self.mrr(tgt_poi_logits, tgt_pois)
        
        self.poi_loss_avg.update(poi_loss)
        for idx, gh_loss in enumerate(gh_losses):
            self.geohash_losses_avg[idx].update(gh_loss)
        self.acc1_avg.update(acc1_val.detach().cpu().item())
        self.acc5_avg.update(acc5_val.detach().cpu().item())
        self.acc10_avg.update(acc10_val.detach().cpu().item())
        self.acc20_avg.update(acc20_val.detach().cpu().item())
        self.mrr_avg.update(mrr_val.detach().cpu().item())

        return total_loss / 4
    
    def on_test_epoch_start(self):
        self.poi_loss_avg.reset()
        self.acc1_avg.reset()
        self.acc5_avg.reset()
        self.acc10_avg.reset()
        self.acc20_avg.reset()
        self.mrr_avg.reset()
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        headers = ["Metric", "Score"]
        table = [
            ["Acc@1", self.acc1_avg.avg],
            ["Acc@5", self.acc5_avg.avg],
            ["Acc@10", self.acc10_avg.avg],
            ["Acc@20", self.acc20_avg.avg],
            ["MRR", self.mrr_avg.avg],
        ]
        print(tabulate(table, headers, tablefmt="rounded_grid"))
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        """
        Optimizer configuration
        """
        if self.optim_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_lr)
        elif self.optim_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': "Val/Loss"
        }



