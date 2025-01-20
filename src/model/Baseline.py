import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import lightning.pytorch as pl

from ..metrics import MRR, AccuracyK


class TrajLSTM(pl.LightningModule):
    
    def __init__(self,
                 num_user:int = None,
                 user_emb_dim:int = 256,
                 num_poi:int = None,
                 poi_emb_dim:int = 256,
                 hidden_dim:int = 512,
                 num_layers:int = 1,
                 lstm_dropout:float = 0,
                 emb_dropout:float = 0,
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


        
        self.user_emb = nn.Embedding(num_embeddings=num_user,
                                     embedding_dim=user_emb_dim,
                                     padding_idx=0)
        self.poi_emb = nn.Embedding(num_embeddings=num_poi,
                                    embedding_dim=poi_emb_dim,
                                    padding_idx=0)
        
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        
        
        self.net = nn.LSTM(input_size=poi_emb_dim + user_emb_dim,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=lstm_dropout)
        
        self.out_projector = nn.Linear(in_features=hidden_dim,
                                       out_features=num_poi)
        
        
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
        
        inputs = torch.cat([poi_embs, user_embs], dim=-1)  # shape (batch, seq_len, poi_emb_dim + user_emb_dim)
        inputs = self.emb_dropout(inputs)
        
        pck_inputs = pack_padded_sequence(
            inputs,
            lengths=orig_lengths,      
            batch_first=True, 
            enforce_sorted=False       # or True if you're sure it's sorted
        )
        
        pck_output, _ = self.net(pck_inputs)  
        
        lstm_out, _ = pad_packed_sequence(pck_output, batch_first=True) # shape (batch, seq_len, hidden_dim)
        
        # out = torch.cat((lstm_out, self.emb_dropout(user_embs)), dim=2) # shape (bathc, seq_len, hidden_dim + user_emb_dim)
        logits = self.out_projector(lstm_out) # shape (batch, seq_len, num_poi)
        
        return logits

    def training_step(self, batch):
        # Training step logic
        """
            Batch contains x, y, original lengths,
            where x and y contain (in order):
                [User ID, POIs, POIs Category, POIs Geohash, POIs Time Slot, POIs Unix Timestamp]
                Each item in the batch has 1 User ID and a sequence of checki-ins (the rest of the items are lists)
            - y, is the shifted version of x by 1 time step (teacher forcing)
            
        """
        x, y, orig_lengths = batch
        user_ids, pois = x[0], x[1] # User ID shape [batch], POIs shape [batch, seq_len]
        seq_len = pois.size(1)
        
        mask = torch.arange(seq_len).expand(len(orig_lengths), seq_len) < orig_lengths.unsqueeze(1)
        # user IDs are the same for all elements in the sequence (trajectory)
        user_ids = user_ids.unsqueeze(1, 2).repeat(1, seq_len) # shape [batch, seq_len]
        user_ids *= mask
        
        logits = self.forward(user_ids, pois)
        
        
        logits = logits.view(-1, self.num_poi)
        true_pois = y[1] # shifted POIs (teacher forcing) shape [batch, seq_len]
        true_pois = true_pois.reshape(-1)
        
        loss = self.loss_fn(logits, true_pois)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
        
        predictions = torch.argmax(logits, dim=-1)  # (batch_size * seq_len)
        valid_mask = true_pois != 0  # Exclude padding from metrics
        
        acc = (predictions[valid_mask] == y[valid_mask]).float().mean()
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)

    def validation_step(self, batch):
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
        user_ids, pois = x[0], x[1] # User ID shape [batch], POIs shape [batch, seq_len]
        seq_len = pois.size(1)
        
        mask = torch.arange(seq_len).expand(len(orig_lengths), seq_len) < orig_lengths.unsqueeze(1)
        user_ids = user_ids.unsqueeze(1, 2).repeat(1, seq_len) # shape [batch, seq_len]
        user_ids *= mask
        
        logits = self.forward(user_ids, pois) # shape (batch, seq_len, num_poi)
        last_logit = logits[:, -1, :].squeeze(1) # shape (batch, num_poi)
        
        true_pois = y[1][0] # take the first POI in the test check-ins shape (batch, 1)
        true_pois = true_pois.reshape(-1)
        
        loss = self.loss_fn(last_logit, true_pois)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        pass
    
    
    def evaluation_step(self, trajectory):
        pass
    

    def configure_optimizers(self):
        """
        Optimizer configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer