
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random


class UserTrajectoryDataset(Dataset):
    """
    This class provides the necessary functionalities for sampling user trajectories
    during training and testing phases. The class provided a custom collate function
    which is used for batching trajectories of different lengths for faster traning.
    """

    def __init__(
        self,
        train_trajectories: pd.DataFrame,
        test_trajectories: pd.DataFrame,
    ) -> None:
        """
        The function recieves the preprocessed train and test user trajectories
        and sets them as class properties.
        
        Args:
            `train_trajectories`: (pandas.DataFrame): 
                The dataframe contating users and a trajectory associated to each user.
            `test_trajectories`: (pandas.DataFrame):
                The dataframe contating test check-ins.
        """
        super().__init__()

        assert (
            train_trajectories.shape[0] == test_trajectories.shape[0]
        ), "For each traning user trajectory there must be one test trajectory."

        train_trajectories = train_trajectories.drop(columns=["Local Time"])
        test_trajectories = test_trajectories.drop(columns=["Local Time"])
        self.train_trajectories = train_trajectories
        self.test_trajectories = test_trajectories
        self.columns = train_trajectories.columns.tolist()

    def __len__(self):
        return self.train_trajectories.shape[0]

    def __getitem__(self, idx):
        """
        This function simply returns the training and test trajectories of
        a user at index `idx`
        
        Args:
            `idx`: (int): 
                Index of the user.
        """
        train_traj = self.train_trajectories.iloc[idx]
        train_items = [torch.tensor(train_traj[col]) for col in self.columns]
        test_traj = self.test_trajectories.iloc[idx]
        test_items = [torch.tensor(test_traj[col]) for col in self.columns]
        return train_items, test_items

    @staticmethod
    def custom_collate(
        batch,
        max_seq_length: int,
        sampling_method: str,
        geohash_precision: list,
        train: bool = True,
    ):
        """
        This function simply returns the training and test trajectories of
        a user at index `idx`
        
        Args:
            `idx`: (int): 
                Index of the user.
        """
        train_batch, test_batch = zip(*batch)

        if len(geohash_precision) == 1:
            users, pois, pois_cat, gh1, ts, ut = zip(*train_batch)
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_ts, tgt_ut = zip(
                *test_batch
            )
            gh2, gh3, tgt_gh2, tgt_gh3 = None, None, None, None
        elif len(geohash_precision) == 2:
            users, pois, pois_cat, gh1, gh2, ts, ut = zip(*train_batch)
            tgt_users, tgt_pois, tgt_pois_cat, tgt_gh1, tgt_gh2, tgt_ts, tgt_ut = zip(
                *test_batch
            )
            gh3, tgt_gh3 = None
        elif len(geohash_precision) == 3:
            users, pois, pois_cat, gh1, gh2, gh3, ts, ut = zip(*train_batch)
            (
                tgt_users,
                tgt_pois,
                tgt_pois_cat,
                tgt_gh1,
                tgt_gh2,
                tgt_gh3,
                tgt_ts,
                tgt_ut,
            ) = zip(*test_batch)
        else:
            raise RuntimeError(
                "The case for more than 3 geohash precisions in not handeled."
            )

        # During the test phase we concatenate the target visits to the
        # training trajectory except the last visit since we are not going to
        # predict any visit after the last visit.
        if not train:
            pois = tuple(
                torch.cat([tr, tgt[:-1]], dim=0) for tr, tgt in zip(pois, tgt_pois)
            )
            pois_cat = tuple(
                torch.cat([tr, tgt[:-1]], dim=0)
                for tr, tgt in zip(pois_cat, tgt_pois_cat)
            )
            gh1 = tuple(
                torch.cat([tr, tgt[:-1]], dim=0) for tr, tgt in zip(gh1, tgt_gh1)
            )
            if gh2 is not None:
                gh2 = tuple(
                    torch.cat([tr, tgt[:-1]], dim=0) for tr, tgt in zip(gh2, tgt_gh2)
                )
            if gh3 is not None:
                gh3 = tuple(
                    torch.cat([tr, tgt[:-1]], dim=0) for tr, tgt in zip(gh3, tgt_gh3)
                )
            ts = tuple(torch.cat([tr, tgt[:-1]], dim=0) for tr, tgt in zip(ts, tgt_ts))
            ut = tuple(torch.cat([tr, tgt[:-1]], dim=0) for tr, tgt in zip(ut, tgt_ut))

        # In the test phase we pad all sequences in the batch to the longest
        # sequence length in the batch since we don't want to remove any
        # information.
        if not train or max_seq_length == -1:
            max_seq_length = np.max([len(item) for item in pois])

        # Helper function to process each sequence based on the sampling method
        def process_sequence(seq):
            L = len(seq)
            if L > max_seq_length:
                if sampling_method == "window":
                    # Sample a continuous interval of length max_seq_length
                    start_idx = random.randint(0, L - max_seq_length)
                    seq = seq[start_idx : start_idx + max_seq_length]
                elif sampling_method == "random":
                    # Sample max_seq_length indices randomly
                    indices = sorted(random.sample(range(L), max_seq_length))
                    seq = [seq[i] for i in indices]

            if not isinstance(seq, torch.Tensor):
                seq = torch.tensor(seq)
            return seq, len(seq)

        processed_pois = [process_sequence(seq) for seq in pois]
        processed_pois_cat = [process_sequence(seq) for seq in pois_cat]
        processed_gh1 = [process_sequence(seq) for seq in gh1]
        if gh2:
            processed_gh2 = [process_sequence(seq) for seq in gh2]
        if gh3:
            processed_gh3 = [process_sequence(seq) for seq in gh3]
        processed_ts = [process_sequence(seq) for seq in ts]
        processed_ut = [process_sequence(seq) for seq in ut]

        pois, pois_lens = zip(*processed_pois)
        pois_cat, _ = zip(*processed_pois_cat)
        gh1, _ = zip(*processed_gh1)
        if gh2:
            gh2, _ = zip(*processed_gh2)
        if gh3:
            gh3, _ = zip(*processed_gh3)
        ts, _ = zip(*processed_ts)
        ut, _ = zip(*processed_ut)

        # Pad sequences to the longest sequence length
        pois = pad_sequence(pois, batch_first=True, padding_value=0)
        pois_cat = pad_sequence(pois_cat, batch_first=True, padding_value=0)
        gh1 = pad_sequence(gh1, batch_first=True, padding_value=0)
        if gh2:
            gh2 = pad_sequence(gh2, batch_first=True, padding_value=0)
        if gh3:
            gh3 = pad_sequence(gh3, batch_first=True, padding_value=0)
        ts = pad_sequence(ts, batch_first=True, padding_value=0)
        ut = pad_sequence(ut, batch_first=True, padding_value=0)

        users = torch.tensor(users)

        ghs = list(item for item in [gh1, gh2, gh3] if item is not None)

        # During training the targets are shifted version of the input
        # During test phase, the targets are unseen data.
        if train:
            # Prepare x and y for training with teacher forcing
            # In some of the batches all of the sequences are shorter than max_seq_length
            orig_lens = torch.tensor(pois_lens)
            if not torch.any(orig_lens == max_seq_length):
                max_seq_length = orig_lens.max().item()

            orig_lens -= 1
            mask = torch.arange(max_seq_length - 1).expand(
                len(orig_lens), max_seq_length - 1
            ) < orig_lens.unsqueeze(1)
            x = (
                users,
                pois[:, :-1] * mask,
                pois_cat[:, :-1] * mask,
                *[gh_[:, :-1] * mask for gh_ in ghs],
                ts[:, :-1] * mask,
                ut[:, :-1] * mask,
            )
            y = (users, pois[:, 1:], pois_cat[:, 1:], *[gh_[:, 1:] for gh_ in ghs])
            return x, y, orig_lens
        else:
            orig_lens = torch.tensor(pois_lens)
            # Input is the complete user trajectories
            x = (users, pois, pois_cat, *ghs, ts, ut)

            tgt_ghs = list(
                item for item in [tgt_gh1, tgt_gh2, tgt_gh3] if item is not None
            )
            y = (
                torch.tensor(tgt_users),
                torch.stack(tgt_pois),
                torch.stack(tgt_pois_cat),
                *[torch.stack(tgt_gh) for tgt_gh in tgt_ghs],
            )
            return x, y, orig_lens