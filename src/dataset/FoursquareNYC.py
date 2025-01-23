import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from math import radians, sin, cos, sqrt, atan2, degrees

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torch_geometric.typing import SparseTensor

from pathlib import Path


import shutil
from datetime import datetime, timedelta
import geohash2 as geohash
from typing import Union
from functools import partial
import random

from ..utils import misc_utils


class UserTrajectoryDataset(Dataset):

    def __init__(
        self,
        train_trajectories: pd.DataFrame,
        test_trajectories: pd.DataFrame,
    ) -> None:
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
                *[torch.stack(tgt_gh) for tgt_gh in tgt_ghs]
            )
            return x, y, orig_lens


class FoursquareNYC(LightningDataModule):
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        batch_size: int = 32,
        num_workers: int = 8,
        user_checkin_tsh: int = (20, np.inf),
        venue_checkin_tsh: int = (10, np.inf),
        num_test_checkins: int = 6,
        geohash_precision: list = [5, 6, 7],
        max_traj_length: list = 64,
        traj_sampling_method: str = "window",
        temporal_graph_jaccard_mult_set: bool = True,
        temporal_graph_jaccard_sim_tsh: float = 0.9,
        spatial_graph_self_loop: bool = False,
        spatial_graph_geohash_precision: int = 6,
        temporal_graph_self_loop: bool = False,
        seed: int = 11,
    ) -> None:

        super().__init__()

        assert (
            spatial_graph_geohash_precision in geohash_precision
        ), "The precision of Geohash used to form the spatial graph must be present in the `geohash_precision` list."

        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.FNYC_dir = self.data_dir.joinpath(Path("FNYC"))
        self.FNYC_dir.mkdir(exist_ok=True)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.DF_COLUMNS = [
            "User ID",
            "Venue ID",
            "Venue Category ID",
            "Venue Category Name",
            "Latitude",
            "Longitude",
            "Timezone Offset",
            "UTC Time",
        ]

        self.num_test_checkins = num_test_checkins
        self.user_checkin_tsh = user_checkin_tsh
        self.venue_checkin_ths = venue_checkin_tsh

        self.geohash_precision = geohash_precision

        self.max_traj_length = max_traj_length
        self.traj_sampling_method = traj_sampling_method

        self.temporal_graph_jaccard_mult_set = temporal_graph_jaccard_mult_set
        self.temporal_graph_jaccard_sim_tsh = temporal_graph_jaccard_sim_tsh
        self.spatial_graph_self_loop = spatial_graph_self_loop
        self.spatial_graph_geohash_precision = spatial_graph_geohash_precision
        self.temporal_graph_self_loop = temporal_graph_self_loop

        self._download_dataset()
        self._load_data()
        self._preprocess_data()

        self._construct_spatial_graph()
        self._construct_temporal_graph()
        self._construct_hierarchical_spatial_graph()

    def prepare_data(self):
        # Download or prepare data if needed
        ...

    def setup(self, stage: str = None):
        # Split the dataset into train/val/test and assign to self variables
        self.dataset = UserTrajectoryDataset(
                self.user_train_trajectories, self.user_test_trajectories
            )
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset
        # if stage == 'predict' or stage is None:
        #     self.predict_dataset = ...

    def train_dataloader(self):
        collate_fn = partial(
            UserTrajectoryDataset.custom_collate,
            max_seq_length=self.max_traj_length,
            sampling_method=self.traj_sampling_method,
            geohash_precision=self.geohash_precision,
            train=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        collate_fn = partial(
            UserTrajectoryDataset.custom_collate,
            max_seq_length=-1,
            sampling_method=self.traj_sampling_method,
            geohash_precision=self.geohash_precision,
            train=False,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        collate_fn = partial(
            UserTrajectoryDataset.custom_collate,
            max_seq_length=-1,
            sampling_method=self.traj_sampling_method,
            geohash_precision=self.geohash_precision,
            train=False,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def _construct_spatial_graph(self):
        """
        Constructs spatial POI-POI graph based on the equality of geohash codes on the specified precision.
        The `spatial_graph_geohash_precision` argument of the class identifies the precision level.
        POIs within the same area have a link.
        """
        geohash_vec = np.array(
            self.poi_trajectories[f"Geohash P{self.spatial_graph_geohash_precision} ID"]
        )

        adj_mat = (geohash_vec[:, None] == geohash_vec).astype(np.int8)
        if not self.spatial_graph_self_loop:
            np.fill_diagonal(adj_mat, 0)  # Remove self-loops

        print(f"Spatial graph sparsity: {self.get_sparsity(adj_mat)}")
        # Add empty row and column at position zero so later we can
        # retrieve the neighbors simply by POI ID
        adj_mat = np.pad(adj_mat, ((1, 0), (1, 0)), mode="constant", constant_values=0)

        self.spatial_graph = torch.tensor(adj_mat)
        # self.spatial_graph = self.convert_to_sparse_tensor(adj_mat)

    def _construct_temporal_graph(self):
        """
        Constructs the temporal POI-POI graph based on the Jaccard similarity of the time slots of the check-ins
        in the specific POI. If two POIs have similar time slots for their check-ins, there will be a
        link between them.
        """
        num_venues = len(self.poi_trajectories["Venue ID"])
        num_timeslots = 56
        M = np.zeros((num_venues, num_timeslots), dtype=int)

        for i, row in enumerate(self.poi_trajectories["Time Slot"]):
            for t in row:
                M[i, t - 1] += 1

        if self.temporal_graph_jaccard_mult_set:
            # Multiset Jaccard Similarity (multiset implementation)
            intersection = M @ M.T
            row_sums = M.sum(axis=1)  # Sum of frequencies for each venue
            union = row_sums[:, None] + row_sums[None, :] - intersection

        else:
            # Standard Set Jaccard Similarity (set implementation)
            # Convert M into a binary matrix (1 if the time slot is present, 0 otherwise)
            M_binary = (M > 0).astype(int)
            intersection = M_binary @ M_binary.T  # Pairwise set intersection counts
            row_sums = M_binary.sum(axis=1)  # Count of unique time slots for each venue
            union = row_sums[:, None] + row_sums[None, :] - intersection

        # Step 2: Compute Jaccard Similarity
        jaccard_similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )

        # Step 3: Threshold the similarity to construct the adjacency matrix
        adj_mat = (jaccard_similarity > self.temporal_graph_jaccard_sim_tsh).astype(
            np.int8
        )
        if not self.temporal_graph_self_loop:
            np.fill_diagonal(adj_mat, 0)  # Remove self-loops

        print(f"Temporal graph sparsity: {self.get_sparsity(adj_mat)}")
        # Add empty row and column at position zero so later we can
        # retrieve the neighbors simply by POI ID
        adj_mat = np.pad(adj_mat, ((1, 0), (1, 0)), mode="constant", constant_values=0)
        # Step 4: Convert to sparse tensor and store
        self.temporal_graph = torch.tensor(adj_mat)
        # self.temporal_graph = self.convert_to_sparse_tensor(adj_mat)

    def _construct_hierarchical_spatial_graph(self):
        """
        Constructs hierarchical mappings from one geohash precision to the next and finally to POIs.
        The levels of the graph start from the lowest geohash precision and forms one layer for each
        geohash precisoin and in the final layer adds POIs.

        """
        df = self.poi_trajectories
        graph = {}

        for i in range(len(self.geohash_precision) - 1):
            parnent_prc = self.geohash_precision[i]
            child_prc = self.geohash_precision[i + 1]

            parent_col = f"Geohash P{parnent_prc} ID"
            child_col = f"Geohash P{child_prc} ID"

            # Group children by their parent geohash
            parent_to_children = (
                df.groupby(parent_col)[child_col]
                .apply(lambda x: list(x.unique()))
                .to_dict()
            )
            graph[f"P{parnent_prc}_to_P{child_prc}"] = parent_to_children

        # Map last precision geohashes to POIs
        graph[f"P{self.geohash_precision[-1]}_to_POI"] = (
            df.groupby(f"Geohash P{self.geohash_precision[-1]} ID")["Venue ID"]
            .apply(lambda x: list(x.unique()))
            .to_dict()
        )

        self.hierarchical_spatial_graph = graph

    def _preprocess_data(self):

        self.df = self.df.drop_duplicates()

        print("Dateset statistics before filtering:")
        self._log_stats(self.df)

        df_flt = self._filter_user_venue(self.df.copy(deep=True))
        df_flt = self._filter_user_venue(df_flt)
        print("Dateset statistics after filtering:")
        self._log_stats(df_flt)

        df_flt = self._reassign_IDs(df_flt)

        # distances = self._calculate_discrepancy_in_locations(df_flt)
        # i = 0
        # for venue, distance in distances:
        #     print(venue, distance)
        #     i += 1
        #     if i == 5:
        #         break
        # i = 0
        # for venue, distance in distances:
        #     if distance > 100.0:
        #         i+=1
        # print('num venues with more than 200 min check-ins', i)

        df_flt = self._replace_locations_with_average(df_flt)

        # distances = self._calculate_discrepancy_in_locations(df_flt)
        # i = 0
        # for venue, distance in distances:
        #     print(venue, distance)
        #     i += 1
        #     if i == 5:
        #         break
        # i = 0
        # for venue, distance in distances:
        #     if distance > 100.0:
        #         i+=1

        # print('num venues with more than 200 min check-ins', i)

        self.STATS = {
            "num_user": df_flt["User ID"].nunique(),
            "num_pois": df_flt["Venue ID"].nunique(),
            "num_poi_cat": df_flt["Venue Category ID"].nunique(),
            "num_time_slots": 56,
        }

        df_flt = self._process_time(df_flt)
        df_flt = self._process_location(df_flt)

        gh_id_keys = [f"Geohash P{prc} ID" for prc in self.geohash_precision]
        stats_gh_keys = [f"num_gh_P{prc}" for prc in self.geohash_precision]

        self.STATS.update(
            {
                **{
                    key: val
                    for key, val in zip(
                        stats_gh_keys,
                        [df_flt[gh_key].nunique() for gh_key in gh_id_keys],
                    )
                },
            }
        )

        # issues = self._check_geohash_consistency(df_flt)
        # print(self.STATS)
        # exit()

        # user_trajectories, poi_trajectories = self._from_trajectories(df_flt)
        user_train_trajectories, user_test_trajectories, poi_trajectories = (
            self._from_trajectories_with_split(df_flt, self.num_test_checkins)
        )

        # # Compute the lengths of the lists in the 'User ID' column
        # lengths = poi_trajectories['User ID'].apply(len)

        # # Calculate the minimum, maximum, and average length
        # min_length = lengths.min()
        # max_length = lengths.max()
        # average_length = lengths.mean()

        # print(f"Minimum length: {min_length}")
        # print(f"Maximum length: {max_length}")
        # print(f"Average length: {average_length}")

        # print(df_flt['Geohash ID'])
        # print(df_flt['Time Slot'].min(), df_flt['Time Slot'].max())

        # print(user_trajectories.shape)
        # print(user_trajectories.iloc[0])

        # print(poi_trajectories.shape)
        # print(poi_trajectories.head())
        # print(poi_trajectories.iloc[0])
        # print(user_trajectories.head()['Geohash ID'])

        # print(self.search_user_venue_visits(user_trajectories, 48, 4))

        self.df_preprocessed = df_flt
        self.user_train_trajectories = user_train_trajectories
        self.user_test_trajectories = user_test_trajectories
        # self.user_trajectories = user_trajectories
        self.poi_trajectories = poi_trajectories
        # memory_usage = self.df_preprocessed.memory_usage(deep=True).sum()
        # print(f"The DataFrame is using {memory_usage / 1024:.2f} KB in memory.")

    def _from_trajectories_with_split(self, df: pd.DataFrame, num_test_checkins: int):
        # Initialize lists for train and test trajectories
        user_train_trajectories = []
        user_test_trajectories = []

        # Store training data for calculating POI trajectories later
        train_checkins = []

        geohash_id_keys = [
            f"Geohash P{precision} ID" for precision in self.geohash_precision
        ]

        for user_id, user_data in df.groupby("User ID", sort=False):
            user_data = user_data.sort_values(
                by="Unix Timestamp"
            )  # Ensure temporal ordering
            train_data = (
                user_data.iloc[:-num_test_checkins]
                if len(user_data) > num_test_checkins
                else user_data.iloc[0:0]
            )
            test_data = (
                user_data.iloc[-num_test_checkins:]
                if len(user_data) >= num_test_checkins
                else user_data
            )

            if not train_data.empty:
                train_checkins.append(
                    train_data
                )  # Collect for POI trajectory calculation
                geohash_id_values = [
                    train_data[id_key].tolist() for id_key in geohash_id_keys
                ]
                geohash_key_values = {
                    key: value for key, value in zip(geohash_id_keys, geohash_id_values)
                }
                user_train_trajectories.append(
                    {
                        "User ID": user_id,
                        "Venue ID": train_data["Venue ID"].tolist(),
                        "Venue Category ID": train_data["Venue Category ID"].tolist(),
                        **geohash_key_values,
                        "Local Time": train_data["Local Time"].tolist(),
                        "Time Slot": train_data["Time Slot"].tolist(),
                        "Unix Timestamp": train_data["Unix Timestamp"].tolist(),
                    }
                )

            if not test_data.empty:
                geohash_id_values = [
                    test_data[id_key].tolist() for id_key in geohash_id_keys
                ]
                geohash_key_values = {
                    key: value for key, value in zip(geohash_id_keys, geohash_id_values)
                }
                user_test_trajectories.append(
                    {
                        "User ID": user_id,
                        "Venue ID": test_data["Venue ID"].tolist(),
                        "Venue Category ID": test_data["Venue Category ID"].tolist(),
                        **geohash_key_values,
                        "Local Time": test_data["Local Time"].tolist(),
                        "Time Slot": test_data["Time Slot"].tolist(),
                        "Unix Timestamp": test_data["Unix Timestamp"].tolist(),
                    }
                )

        user_train_trajectories_df = pd.DataFrame(user_train_trajectories)
        user_test_trajectories_df = pd.DataFrame(user_test_trajectories)

        # Combine all training data into a single DataFrame
        train_data_only = pd.concat(train_checkins, ignore_index=True)

        # Create POI trajectories from training data

        poi_trajectories = (
            train_data_only.groupby("Venue ID", sort=False)
            .agg(
                {
                    **{
                        key: value
                        for key, value in zip(
                            geohash_id_keys, ["first"] * len(geohash_id_keys)
                        )
                    },
                    "Venue Category ID": "first",
                    "User ID": list,
                    "Local Time": list,
                    "Time Slot": list,
                    "Unix Timestamp": list,
                }
            )
            .reset_index()
        )
        poi_trajectories = poi_trajectories.sort_values(by=["Venue ID"])

        return user_train_trajectories_df, user_test_trajectories_df, poi_trajectories

    def _process_location(self, df: pd.DataFrame):

        for precision in self.geohash_precision:
            df[f"Geohash P{precision}"] = df.apply(
                lambda row: geohash.encode(
                    row["Latitude"], row["Longitude"], precision=precision
                ),
                axis=1,
            )
        df = df.drop(columns=["Latitude", "Longitude"])

        self.GEOHASH_TO_ID_MAP = {}
        self.ID_TO_GEOHASH_MAP = {}
        # Create a mapping dictionary for geohash to integer ID
        for precision in self.geohash_precision:
            geohash_key = f"Geohash P{precision}"
            geohash_id_key = f"Geohash P{precision} ID"

            unique_geohashes = df[geohash_key].unique()
            geohash_to_id = {
                gh: idx for idx, gh in enumerate(unique_geohashes, start=1)
            }  # Direct map for ID
            id_to_geohash = {
                idx: gh for gh, idx in geohash_to_id.items()
            }  # Reverse map for recovery
            df[geohash_id_key] = df[geohash_key].map(geohash_to_id)
            df = df.drop(columns=[geohash_key])

            self.GEOHASH_TO_ID_MAP[geohash_key] = geohash_to_id
            self.ID_TO_GEOHASH_MAP[geohash_key] = id_to_geohash

        return df

    def _check_geohash_consistency(self, df: pd.DataFrame):
        issues = {}
        for precision in self.geohash_precision:
            column_name = f"Geohash P{precision} ID"
            # Group by Venue ID and count unique geohashes per group
            inconsistent_venues = df.groupby("Venue ID")[column_name].nunique()
            # Find Venue IDs with more than one unique geohash
            problematic = inconsistent_venues[inconsistent_venues > 1]
            if not problematic.empty:
                issues[precision] = problematic.index.tolist()
        return issues

    def _process_time(self, df: pd.DataFrame):

        def calculate_local_time(utc_time, timezone_offset):
            # Parse the UTC time string
            utc_datetime = datetime.strptime(utc_time, "%a %b %d %H:%M:%S +0000 %Y")
            # Add the timezone offset
            local_datetime = utc_datetime + timedelta(minutes=timezone_offset)
            return local_datetime

        def local_time_to_week_slot(df) -> int:
            # Extract the day of the week (Monday=0, Sunday=6) and hour
            df["Day of Week"] = df["Local Time"].dt.weekday
            df["Hour"] = df["Local Time"].dt.hour

            # Calculate the time slot (each day has 8 slots, each slot is 3 hours)
            # Slot formula: (Day of Week * 8) + (Hour // 3)
            df["Time Slot"] = (df["Day of Week"] * 8) + (df["Hour"] // 3) + 1

            # Drop intermediate columns if not needed
            df = df.drop(columns=["Day of Week", "Hour"])

            return df

        df["Local Time"] = df.apply(
            lambda row: calculate_local_time(row["UTC Time"], row["Timezone Offset"]),
            axis=1,
        )
        df["Unix Timestamp"] = df["Local Time"].apply(lambda x: int(x.timestamp()))
        df = local_time_to_week_slot(df)
        # min_timestamp = df['Unix Timestamp'].min()
        # max_timestamp = df['Unix Timestamp'].max()
        # df['Normalized Timestamp'] = (df['Unix Timestamp'] - min_timestamp) / (max_timestamp - min_timestamp)
        df = df.sort_values(by=["User ID", "Unix Timestamp"])
        df = df.drop(columns=["Timezone Offset", "UTC Time"])

        return df

    def _reassign_IDs(self, df: pd.DataFrame):
        df["User ID"] = pd.factorize(df["User ID"])[0] + 1
        venue_id_mapping = {
            id_: idx for idx, id_ in enumerate(df["Venue ID"].unique(), start=1)
        }
        venue_category_id_mapping = {
            id_: idx
            for idx, id_ in enumerate(df["Venue Category ID"].unique(), start=1)
        }
        df["Venue ID"] = df["Venue ID"].map(venue_id_mapping)
        df["Venue Category ID"] = df["Venue Category ID"].map(venue_category_id_mapping)
        venue_category_id_name_mapping = df[
            ["Venue Category ID", "Venue Category Name"]
        ].drop_duplicates()
        venue_category_id_name_mapping = dict(
            zip(
                venue_category_id_name_mapping["Venue Category ID"],
                venue_category_id_name_mapping["Venue Category Name"],
            )
        )

        df = df.drop(columns=["Venue Category Name"])
        self.VENUE_CAT_ID_MAP = venue_category_id_name_mapping
        return df

    def _filter_user_venue(self, df: pd.DataFrame):
        num_user_checkins = df["User ID"].value_counts()
        num_venue_checkins = df["Venue ID"].value_counts()
        # num_venue_users = self.df.groupby("Venue ID")["User ID"].nunique()
        # venues_with_enough_users = num_venue_users[(num_venue_users >= self.venue_checkin_ths[0]) & (num_venue_users <= self.venue_checkin_ths[1])].index
        venues_with_enough_users = num_venue_checkins[
            (num_venue_checkins >= self.venue_checkin_ths[0])
            & (num_venue_checkins <= self.venue_checkin_ths[1])
        ].index
        df = df[df["Venue ID"].isin(venues_with_enough_users)]

        num_user_checkins = df["User ID"].value_counts()
        num_venue_checkins = df["Venue ID"].value_counts()

        filter_idxs = num_user_checkins[
            (num_user_checkins >= self.user_checkin_tsh[0])
            & (num_user_checkins <= self.user_checkin_tsh[1])
        ].index
        df = df[df["User ID"].isin(filter_idxs)]
        df = df.sort_values(by=["User ID"])
        return df

    def _load_data(self):
        df = pd.read_csv(
            self.raw_file_path, sep="\t", encoding="latin-1", names=self.DF_COLUMNS
        )
        self.df = df.sort_values(by=["User ID"])
        # column_null_counts = df.isnull().sum()  # Count of 'null' values in each column
        # print(f"Number of rows: {df.shape[0]} | Number of columns: {df.shape[1]} ")
        # print("\nSummary of 'null' values in each column:")
        # print(column_null_counts)

    def _log_stats(self, df: pd.DataFrame):
        num_user_checkins = df["User ID"].value_counts()
        num_venue_checkins = df["Venue ID"].value_counts()
        print(f"Number of records: {df.shape[0]}")
        print(
            f"Number of users: {df['User ID'].nunique()}, with min = {num_user_checkins.min()}, max = {num_user_checkins.max()}, and avg: {num_user_checkins.mean()}"
        )
        print(
            f"Number of venues: {df['Venue ID'].nunique()}, with min = {num_venue_checkins.min()}, max = {num_venue_checkins.max()}, and avg: {num_venue_checkins.mean()}"
        )
        print(f"Number of venue categories: {df['Venue Category ID'].nunique()}")

    def _download_dataset(self):
        zip_file_path = self.FNYC_dir.joinpath(Path("dataset.zip"))
        if zip_file_path.exists():
            print("Dataset already downloaded!")
        else:
            try:
                url = "http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
                print("Starting download...")
                misc_utils.download_file_fast(url, zip_file_path.absolute())

            except Exception as e:
                raise RuntimeError(f"Failed to download the dataset: {e}")

        self.raw_files_dir = self.FNYC_dir.joinpath("raw_files")
        if self.raw_files_dir.exists():
            print("Dataset already extracted!")
        else:
            try:
                print("Extracting the file...")
                self.raw_files_dir.mkdir(exist_ok=False)
                misc_utils.extract_zip_multithreaded(
                    zip_file_path.absolute(), self.raw_files_dir.absolute()
                )
                print(
                    f"Extraction completed. Files are available in: {self.raw_files_dir}"
                )
            except Exception as e:
                shutil.rmtree(self.raw_files_dir)
                raise RuntimeError(f"Failed to extraxt the dataset: {e}")

        self.raw_file_path = self.raw_files_dir.joinpath(
            "dataset_tsmc2014/dataset_TSMC2014_NYC.txt"
        )

    def search_user_venue_visits(self, df, user_id, venue_id):
        user_row = df[df["User ID"] == user_id]

        if user_row.empty:
            return f"User ID {user_id} not found in the DataFrame."
        user_row = user_row.iloc[0]

        visit_indices = [i for i, v in enumerate(user_row["Venue ID"]) if v == venue_id]

        if not visit_indices:
            return f"Venue ID {venue_id} not visited by User ID {user_id}."

        visit_times = [user_row["Local Time"][i] for i in visit_indices]

        result = pd.DataFrame(
            {"Venue ID": [venue_id] * len(visit_indices), "Local Time": visit_times}
        )

        return result

    def get_sparsity(self, mat):
        total_elements = mat.size
        num_zeroes = np.sum(mat == 0)
        percentage_zeroes = (num_zeroes / total_elements) * 100
        return percentage_zeroes

    def convert_to_sparse_tensor(sefl, mat):
        # Convert the matrix to a sparse tensor
        row, col = np.where(mat)  # Find indices of non-zero elements
        sparse_tensor = SparseTensor(
            row=torch.tensor(row),
            col=torch.tensor(col),
            value=torch.tensor(mat[row, col]),
        )
        return sparse_tensor

    def plot_distribution(self, data, x_label="Value", y_label="Frequency"):
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins="auto", edgecolor="black", alpha=0.7)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Distribution Plot")
        plt.show()

    def _collect_unique_venue_locations(self, df: pd.DataFrame):
        # Group by 'Venue ID' and aggregate unique locations (Latitude, Longitude)
        venue_locations = (
            df.groupby("Venue ID")
            .apply(
                lambda group: group[["Latitude", "Longitude"]]
                .drop_duplicates()
                .values.tolist()
            )
            .to_dict()
        )
        return venue_locations

    def _replace_locations_with_average(self, df: pd.DataFrame):
        """
        This method is generated by ChatGPT.
        This method finds all locations assigned to a POI and takes their average.
        """

        def calculate_average_location(locations):
            """
            Calculate the average latitude and longitude for a list of locations using spherical coordinates.

            Args:
                locations (list): A list of tuples representing (latitude, longitude).

            Returns:
                tuple: The average (latitude, longitude) in decimal degrees.
            """
            x = y = z = 0.0

            for lat, lon in locations:
                lat_rad = radians(lat)
                lon_rad = radians(lon)
                x += cos(lat_rad) * cos(lon_rad)
                y += cos(lat_rad) * sin(lon_rad)
                z += sin(lat_rad)

            total = len(locations)
            x /= total
            y /= total
            z /= total

            lon_avg = atan2(y, x)
            hyp = sqrt(x * x + y * y)
            lat_avg = atan2(z, hyp)

            return degrees(lat_avg), degrees(lon_avg)

        # Collect unique locations for each venue
        venue_locations = self._collect_unique_venue_locations(df)

        # Calculate average location for each venue
        average_locations = {
            venue: calculate_average_location(locations)
            for venue, locations in venue_locations.items()
        }

        # Replace latitude and longitude in the original dataframe
        df["Latitude"] = df["Venue ID"].map(lambda venue: average_locations[venue][0])
        df["Longitude"] = df["Venue ID"].map(lambda venue: average_locations[venue][1])

        return df

    def _calculate_discrepancy_in_locations(self, df: pd.DataFrame):
        """
        This method is generate by ChatGPT.
        It first findes the different locations assigned to a POI in the dataset and
        then finds the maximum distance between those locations. The distance is calculated
        based on Latitude and Longitude of the locations and is Haversine Distance in meters.
        """

        def haversine_distance(loc1, loc2):
            """
            Calculate the great-circle distance between two points on the Earth using the Haversine formula.

            Args:
                loc1 (tuple): (latitude, longitude) of the first location in decimal degrees.
                loc2 (tuple): (latitude, longitude) of the second location in decimal degrees.

            Returns:
                float: Distance between the two points in kilometers.
            """
            # Radius of the Earth in kilometers
            R = 6371.0

            # Convert latitude and longitude from degrees to radians
            lat1, lon1 = radians(loc1[0]), radians(loc1[1])
            lat2, lon2 = radians(loc2[0]), radians(loc2[1])

            # Differences in coordinates
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            # Haversine formula
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            # Distance
            return R * c * 1000

        def calculate_longest_distance(venue_locations):
            """
            Calculate the longest great-circle distance between different locations assigned to each unique venue.

            Args:
                venue_locations (dict): A dictionary where keys are Venue IDs and values are lists of unique locations (latitude, longitude).

            Returns:
                dict: A dictionary where keys are Venue IDs and values are the longest great-circle distance (if more than one location exists).
            """
            longest_distances = {}

            for venue, locations in venue_locations.items():
                if len(locations) > 1:
                    # Calculate all pairwise distances using the Haversine formula
                    distances = [
                        haversine_distance(loc1, loc2)
                        for loc1, loc2 in combinations(locations, 2)
                    ]
                    longest_distances[venue] = max(distances)
                else:
                    longest_distances[venue] = 0  # No distance if only one location

            return longest_distances

        def get_sorted_venues_by_distance(longest_distances):
            """
            Sort venues by their longest distances in descending order.

            Args:
                longest_distances (dict): A dictionary where keys are Venue IDs and values are the longest Euclidean distances.

            Returns:
                list: A list of tuples sorted by longest distance in descending order (Venue ID, Longest Distance).
            """
            return sorted(longest_distances.items(), key=lambda x: x[1], reverse=True)

        unique_venue_locations = self._collect_unique_venue_locations(df)
        longest_distances = calculate_longest_distance(unique_venue_locations)
        sorted_venues = get_sorted_venues_by_distance(longest_distances)
        return sorted_venues
