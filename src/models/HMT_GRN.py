import torch
import torch.nn as nn
import numpy as np
import torch_geometric.nn as tgnn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pytorch_lightning as pl

from ..metrics import MRR, AccuracyK
from ..dataset import FoursquareNYC

from typing import List


class HMT_GRN(pl.LightningModule):

    def __init__(
        self,
        dataset: FoursquareNYC,
        user_emb_dim: int = 256,
        poi_emb_dim: int = 512,
        poi_cat_emb_dim: int = 128,
        gh_emb_dim: int = 512,
        ts_emb_dim: int = 128,
        hidden_dim: int = 512,
        emb_switch: List[bool] = [
            True,
            True,
            False,
            False,
        ],  # [user_emb, poi_emb, poi_cat_emb, ts_emb]
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0,
        emb_dropout: float = 0.5,
        num_GAT_heads: int = 4,
        GAT_dropout: float = 0.0,
        task_loss_coefficients: List[float] = [1., 1., 1., 1.],
        optim_lr: float = 1e-4,
        optim_type: str = "adamw",
    ) -> None:

        super().__init__()

        assert (
            emb_switch[0] and emb_switch[1]
        ), "User embeddings and POI embeddings must be used and can't be turned off!"
        
        
        self.hierarchical_spatial_graph = dataset.hierarchical_spatial_graph
        self.register_buffer("temporal_graph", dataset.temporal_graph)
        self.register_buffer("spatial_graph", dataset.spatial_graph)
        
        
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
        self.poi_cat_emb_dim = poi_cat_emb_dim
        self.gh_emb_dim = gh_emb_dim
        self.ts_emb_dim = ts_emb_dim
        self.hidden_dim = hidden_dim

        self.emb_switch = emb_switch

        self.aggregated_emb_dim = np.sum(
            [user_emb_dim, poi_emb_dim, poi_cat_emb_dim, ts_emb_dim]
            * np.array(emb_switch).astype(np.int8)
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

        self.temporalGAT = tgnn.GATConv(
            poi_emb_dim,
            poi_emb_dim,
            heads=num_GAT_heads,
            concat=False,
            dropout=GAT_dropout,
            add_self_loops=False
        )
        self.spatialGAT = tgnn.GATConv(
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
            in_features=hidden_dim, out_features=self.num_pois
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
        poi_embs = self.poi_emb(pois)
        poi_cat_embs = self.poi_cat_emb(pois_cat) if self.emb_switch[2] else None
        ts_embs = self.ts_emb(ts) if self.emb_switch[3] else None
        gh_embs = [
            # self.gh_embeddings[idx](ghs[idx]) if ghs[idx] is not None else None
            self.gh_embeddings[idx](ghs[idx])
            for idx in range(len(self.geohash_precision))
        ]

        
        # Extract the neighbors of the nodes in the batch from spatial and temporal graphs
        # The resulting tensors have shape: (batch, seq_len, num_pois)
        # The third dimension contains binary vector representing neighbors IDs.
        pois_spatial_neighbors = self.spatial_graph[pois]
        pois_temporal_neighbors = self.temporal_graph[pois]
        
        # We create two new tensors to store the spatially and temporally attended POI embeddings
        spatially_attended_poi_embs = torch.zeros_like(poi_embs, dtype=poi_embs.dtype, device=poi_embs.device)
        temporally_attended_poi_embs = torch.zeros_like(poi_embs, dtype=poi_embs.dtype, device=poi_embs.device)
        for batch_index in range(batch_size):
            for seq_index in range(seq_len):
                POI = pois[batch_index, seq_index]
                POI_emb = poi_embs[batch_index, seq_index]
                if POI != 0: # skip padding tokens
                    POI_sp_nbs = torch.nonzero(pois_spatial_neighbors[batch_index, seq_index]).squeeze(1) # shape (num_neighbors)
                    POI_tmp_nbs = torch.nonzero(pois_temporal_neighbors[batch_index, seq_index]).squeeze(1) # shape (num_neighbors)
                    POI_sp_nbs_embs = self.poi_emb(POI_sp_nbs)
                    POI_tmp_nbs_embs = self.poi_emb(POI_tmp_nbs)
                    
                    # POI_sp_nbs_embs = self.emb_dropout(POI_sp_nbs_embs)
                    # POI_tmp_nbs_embs = self.emb_dropout(POI_tmp_nbs_embs)
                    
                    POI_sp_nbs_embs = torch.cat([POI_emb.unsqueeze(0), POI_sp_nbs_embs], dim=0) # shape (num_neighbors+1)
                    POI_tmp_nbs_embs = torch.cat([POI_emb.unsqueeze(0), POI_tmp_nbs_embs], dim=0) # shape (num_neighbors+1)
                    
                    # Cunstruct the COO format edge indecies for GATConv input
                    spatial_edge_index = torch.stack([
                        torch.zeros(len(POI_sp_nbs), dtype=torch.long),  # POI -> neighbors
                        torch.arange(1, len(POI_sp_nbs) + 1, dtype=torch.long)  # neighbors -> POI
                    ], dim=0).to(POI.device)
                    
                    # Since the graph is undirected we have to add edges from neighbors to POI
                    sp_reversed_edges = spatial_edge_index.flip(0)
                    spatial_edge_index = torch.cat([spatial_edge_index, sp_reversed_edges], dim=1)
                    # Add self-loop for POI
                    spatial_edge_index = torch.cat([torch.tensor([[0], [0]]).to(POI.device), spatial_edge_index], dim=1)
                    
                    temporal_edge_index = torch.stack([
                        torch.zeros(len(POI_tmp_nbs), dtype=torch.long),  # POI -> neighbors
                        torch.arange(1, len(POI_tmp_nbs) + 1, dtype=torch.long)  # neighbors -> POI
                    ], dim=0).to(POI.device)
                    
                    tp_reversed_edges = temporal_edge_index.flip(0)
                    temporal_edge_index = torch.cat([temporal_edge_index, tp_reversed_edges], dim=1)
                    temporal_edge_index = torch.cat([torch.tensor([[0], [0]]).to(POI.device), temporal_edge_index], dim=1)
                    
                    POI_spatialy_attended = self.spatialGAT(POI_sp_nbs_embs, spatial_edge_index)[0].squeeze()
                    POI_temporally_attended = self.temporalGAT(POI_tmp_nbs_embs, temporal_edge_index)[0].squeeze()
                    
                    spatially_attended_poi_embs[batch_index, seq_index] = POI_spatialy_attended
                    temporally_attended_poi_embs[batch_index, seq_index] = POI_temporally_attended
                else:
                    spatially_attended_poi_embs[batch_index, seq_index] = POI_emb
                    temporally_attended_poi_embs[batch_index, seq_index] = POI_emb

        
        final_poi_emb = self.attended_poi_projector(
            torch.cat([poi_embs, spatially_attended_poi_embs, temporally_attended_poi_embs], dim=-1)
        )
                    
        lstm_inputs = torch.cat(
            [final_poi_emb, user_embs], dim=-1
        )  # shape (batch, seq_len, poi_emb_dim + user_emb_dim)
        if poi_cat_embs:
            # shape (batch, seq_len, poi_emb_dim + user_emb_dim + poi_cat_emb_dim)
            lstm_inputs = torch.cat([lstm_inputs, poi_cat_embs], dim=-1) 
        if ts_embs:
            # shape (batch, seq_len, poi_emb_dim + user_emb_dim + poi_cat_emb_dim + ts_emb_dim)
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

        poi_logits = self.poi_final_projector(lstm_out)
        
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
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1 = y
            gh2, gh3, tgt_gh2, tgt_gh3 = None, None, None, None
        elif len(self.geohash_precision) == 2:
            users, pois, pois_cat, gh1, gh2, ts, ut = x
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2 = y
            gh3, tgt_gh3 = None
        elif len(self.geohash_precision) == 3:
            users, pois, pois_cat, gh1, gh2, gh3, ts, ut = x
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2, tgt_gh3 = y
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
        
        self.log("Train/Total Loss", total_loss, on_epoch=True, reduce_fx="mean", prog_bar=True)
        self.log("Train/POI Loss", poi_loss, on_epoch=True, reduce_fx="mean", prog_bar=False)
        for idx in range(len(self.geohash_precision)):
            self.log(f"Train/Geohash P{self.geohash_precision[idx]} Loss", gh_losses[idx], on_epoch=True, reduce_fx="mean", prog_bar=False) 
        
        return total_loss / 4

        # self.log("Train/Loss", loss, on_epoch=True, reduce_fx="mean", prog_bar=True)
        # return loss

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
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1 = y
            gh2, gh3, tgt_gh2, tgt_gh3 = None, None, None, None
        elif len(self.geohash_precision) == 2:
            users, pois, pois_cat, gh1, gh2, ts, ut = x
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2 = y
            gh3, tgt_gh3 = None
        elif len(self.geohash_precision) == 3:
            users, pois, pois_cat, gh1, gh2, gh3, ts, ut = x
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2, tgt_gh3 = y
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
        
        # poi_logits = poi_logits.view(-1, self.num_pois)
        # ghs_logits = [
        #     ghs_logits[idx].view(-1, self.num_ghs[idx])
        #     for idx in range(len(self.geohash_precision))
        # ]

        # For prediction we only use the logits related to the last prediction of the model
        last_valid_indices = orig_lengths - 1
        batch_indices = torch.arange(batch_size, device=device)
        
        last_poi_logits = poi_logits[
            batch_indices, last_valid_indices, :
        ]  # shape (batch, num_poi)
        
        last_ghs_logits = [
            ghs_logits[idx][batch_indices, last_valid_indices, :]
            for idx in range(len(self.geohash_precision))
        ] # shape (num_geohash_precisions, (batch, num_gh@P))
        
        last_poi_logits = last_poi_logits.view(-1, self.num_pois)
        last_ghs_logits = [
            last_ghs_logits[idx].view(-1, self.num_ghs[idx])
            for idx in range(len(self.geohash_precision))
        ]
        
        tgt_pois = tgt_pois[:, 0] # take the first POI in the test check-ins, shape (batch)
        tgt_ghs = [
            tgt_gh[:, 0] for tgt_gh in [tgt_gh1, tgt_gh2, tgt_gh3] if tgt_gh is not None
        ] # take the first Geohash in the test check-ins for each precision, shape (batch)

        poi_loss = self.loss_fn(last_poi_logits, tgt_pois.reshape(-1))
        gh_losses = [
            self.loss_fn(last_ghs_logits[idx], tgt_ghs[idx].reshape(-1))
            for idx in range(len(self.geohash_precision))
        ]
        
        poi_loss_weighted = poi_loss * self.task_loss_coefficients[0]
        gh_losses_weighted = [
            gh_losses[idx] * self.task_loss_coefficients[idx + 1]
            for idx in range(len(self.geohash_precision))
        ]
        
        total_loss = poi_loss_weighted + sum(gh_losses_weighted)

        acc1 = self.acc1(last_poi_logits, tgt_pois)
        acc5 = self.acc5(last_poi_logits, tgt_pois)
        acc10 = self.acc10(last_poi_logits, tgt_pois)
        acc20 = self.acc20(last_poi_logits, tgt_pois)
        mrr = self.mrr(last_poi_logits, tgt_pois)
        
        self.log("Val/Total Loss", total_loss, on_epoch=True, reduce_fx="mean", prog_bar=True)
        self.log("Val/POI Loss", poi_loss, on_epoch=True, reduce_fx="mean", prog_bar=False)
        for idx in range(len(self.geohash_precision)):
            self.log(f"Val/Geohash P{self.geohash_precision[idx]} Loss", gh_losses[idx], on_epoch=True, reduce_fx="mean", prog_bar=False)
        
        self.log("Val/Acc@1", acc1, on_epoch=True, reduce_fx="mean", prog_bar=False)
        self.log("Val/Acc@5", acc5, on_epoch=True, reduce_fx="mean", prog_bar=False)
        self.log("Val/Acc@10", acc10, on_epoch=True, reduce_fx="mean", prog_bar=False)
        self.log("Val/Acc@20", acc20, on_epoch=True, reduce_fx="mean", prog_bar=False)
        self.log("Val/MRR", mrr, on_epoch=True, reduce_fx="mean", prog_bar=False)
        
        return total_loss / 4

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



