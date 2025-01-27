import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from itertools import combinations
from math import radians, sin, cos, sqrt, atan2, degrees

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .TrajectoryDataset import UserTrajectoryDataset


from torch_geometric.typing import SparseTensor
from torch_geometric.utils import dense_to_sparse, add_self_loops

from pathlib import Path


import shutil
from datetime import datetime, timedelta
import geohash2 as geohash
from typing import Union, List, Tuple
from functools import partial


from ..utils import misc_utils




class FoursquareNYC(LightningDataModule):
    """
    This class is responsible for downloading, loading, filtering, and preprocessing the 
    Foursqure NYC dataset. The class provides dataloader for training, validation, and
    testing phases. Most of the operations are done according to the specifications in
    this papaer: https://dl.acm.org/doi/pdf/10.1145/3477495.3531989
    However since the datasets used in the experiments of this paper differs from NYC
    dataset I change most of the hyperparameters.

    """
    def __init__(
        self,
        data_dir: Path = Path("./data").absolute(),
        batch_size: int = 32,
        num_workers: int = 8,
        user_checkin_tsh: Tuple[int, int] = (20, np.inf),
        venue_checkin_tsh: Tuple[int, int] = (10, np.inf),
        num_test_checkins: int = 6,
        geohash_precision: List[int] = [5, 6, 7],
        # geohash_precision: list = [6],
        max_traj_length: list = 64,
        traj_sampling_method: str = "window",
        temporal_graph_jaccard_mult_set: bool = True,
        temporal_graph_jaccard_sim_tsh: float = 0.9,
        spatial_graph_self_loop: bool = False,
        spatial_graph_geohash_precision: int = 6,
        temporal_graph_self_loop: bool = False,
        plot_stats: bool = False,
        seed: int = 11,
    ) -> None:
        """
        The `init` method carries out all the necessary operations needed for downloading,
        loading, preprocessing, and plotting the statistics of the datset.
        
        Args:
            `data_dir`: (Path): The directory to save and extract the downloaded dataset.
            `batch_size`: (int): Batch size used by data loaders.
            `num_workers`: (int): Number of workers used by data loaders.
            `user_checkin_tsh` (Tuple[int, int]): 
                The threshold used for clipping user trajectories.
            `venue_checkin_tsh` (Tuple[int, int]): 
                The threshold used for clipping poi trajectories.
            `num_test_checkins` (int): Number of check-ins splited for testing.
            `geohash_precision` (List[int]): Precision levels used for Geohash encoding.
            `max_traj_length` (int): 
                The length used to sliced tranjectories in batches during training.
            `traj_sampling_method` (str):
                The method used by collate function to sample sub trajectories during training.
            `temporal_graph_jaccard_mult_set`(bool):
                Whether to use set formulation of Jaccard similarity or multiset formulation.
                In set formulation we neglect repetition while in multiset formulation we count
                repetition of the items in the set. Note that there is no multiset formulation
                of Jaccard similarity! I just made up one version to see if it works better!
            `temporal_graph_jaccard_sim_tsh` (float):
                The cut off threshold of Jaccard similarity used to construct edges in 
                temporal graph.
            `spatial_graph_self_loop` (bool): Whether to add self loop to Spatial Graph.
            `temporal_graph_self_loop` (bool): Whether to add self loop to Temporal Graph.
            `plot_stats` (bool): 
                Wether to plot datset stast usign matplotlob or do simple logging.
        
        """

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

        self.plot_stats = plot_stats

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

    def train_dataloader(self):
        collate_fn = partial(
            UserTrajectoryDataset.custom_collate,
            max_seq_length=self.max_traj_length,
            sampling_method=self.traj_sampling_method,
            geohash_precision=self.geohash_precision,
            train=True,
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

    def get_spatial_graph_edge_indices(
        self, self_loop=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method constructs the edge indices matrix in COO format using the
        adjacency matrix of the Spatial Graph.

        Args:
            `self_loop`: Whether to add self loop to the edge list or not.
        Returns:
            Tuple[Tensor, Tensor]: edge indices and attributes.
        """
        edge_index, edge_attr = dense_to_sparse(self.spatial_graph)
        if self_loop:
            if not torch.diag(self.spatial_graph).sum() == self.STATS["num_pois"]:
                print("Adding self loop to the Spatial adjacency matrix!")
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr=edge_attr, fill_value=1.0
                )

        return edge_index, edge_attr

    def get_temporal_graph_edge_indices(
        self, self_loop=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method constructs the edge indices matrix in COO format using the
        adjacency matrix of the Temporal Graph.

        Args:
            `self_loop`: Whether to add self loop to the edge list or not.
        Returns:
            Tuple[Tensor, Tensor]: edge indices and attributes.
        """
        edge_index, edge_attr = dense_to_sparse(self.temporal_graph)
        if self_loop:
            if not torch.diag(self.temporal_graph).sum() == self.STATS["num_pois"]:
                print("Adding self loop to the Temporal adjacency matrix!")
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr=edge_attr, fill_value=1.0
                )
        return edge_index, edge_attr

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

        # Compute Jaccard Similarity
        jaccard_similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )

        # Threshold the similarity to construct the adjacency matrix
        adj_mat = (jaccard_similarity > self.temporal_graph_jaccard_sim_tsh).astype(
            np.int8
        )
        if not self.temporal_graph_self_loop:
            np.fill_diagonal(adj_mat, 0)  # Remove self-loops
        else:
            np.fill_diagonal(adj_mat, 1)

        print(f"Temporal graph sparsity: {self.get_sparsity(adj_mat)}")
        # Add empty row and column at position zero so later we can
        # retrieve the neighbors simply by POI ID
        adj_mat = np.pad(adj_mat, ((1, 0), (1, 0)), mode="constant", constant_values=0)
        # Convert to sparse tensor and store
        self.temporal_graph = torch.tensor(adj_mat)

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

        if self.plot_stats:
            unique_vens_locs_counts = df_flt.groupby("Venue ID")[
                ["Latitude", "Longitude"]
            ].apply(lambda group: group.drop_duplicates().shape[0])
            values, frequencies = np.unique(unique_vens_locs_counts, return_counts=True)
            plt.figure(figsize=(8, 5))
            plt.bar(values, frequencies, width=0.8, edgecolor="black", alpha=0.7)
            plt.xlabel("Number of Locations Associated to a Venue")
            plt.ylabel("Frequency")
            plt.title("Distribution of Number of Different Locations for Venues")
            plt.xticks(
                np.arange(0, unique_vens_locs_counts.max() + 1)
            )  # Tick marks on integers
            plt.yscale("log")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()
        if self.plot_stats:
            distances = self._calculate_discrepancy_in_locations(df_flt)
            ids, distance_values = zip(*distances)
            distance_values = list(filter(lambda x: x != 0, distance_values))
            x_values = range(1, len(distance_values) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(
                x_values, distance_values, linestyle="-", label="Distance (meters)"
            )
            plt.xlabel("Sorted Venues")
            plt.ylabel("Distance in Meters")
            plt.title(
                "Distance Between Farthest Pair of Locations for Venues with Multiple Locations"
            )
            plt.yscale("log")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()

        # We replace the location of venues with more than one location with the average location
        df_flt = self._replace_locations_with_average(df_flt)

        self.STATS = {
            "num_user": df_flt["User ID"].nunique(),
            "num_pois": df_flt["Venue ID"].nunique(),
            "num_poi_cat": df_flt["Venue Category ID"].nunique(),
            "num_time_slots": 56,
        }

        df_flt = self._process_time(df_flt)
        if self.plot_stats:
            value_counts = df_flt["Time Slot"].value_counts().sort_index()
            plt.figure(figsize=(8, 5))
            plt.bar(
                value_counts.index, value_counts.values, width=0.8, edgecolor="black"
            )
            plt.xlabel("Time Slot")
            plt.ylabel("Frequency")
            plt.title("Distribution of Time Slot Values", fontsize=14)
            plt.xticks(range(1, 57), fontsize=5)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.show()

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

        user_train_trajectories, user_test_trajectories, poi_trajectories = (
            self._from_trajectories_with_split(df_flt, self.num_test_checkins)
        )

        self.df_preprocessed = df_flt
        self.user_train_trajectories = user_train_trajectories
        self.user_test_trajectories = user_test_trajectories
        self.poi_trajectories = poi_trajectories

        if self.plot_stats:
            col_titles = ["Users", "POIs", "POI Categories", "Test Check-ins"]
            for prc in self.geohash_precision:
                col_titles.append(f"Geohash P{prc}")

            row_titles = ["Count"]
            tbl_data = [
                [
                    self.STATS["num_user"],
                    self.STATS["num_pois"],
                    self.STATS["num_poi_cat"],
                    self.num_test_checkins,
                ]
            ]
            for mkey in stats_gh_keys:
                tbl_data[0].append(self.STATS[mkey])
            misc_utils.plot_plt_table(
                main_title="Final Preprocessed Data Stats",
                column_titles=col_titles,
                row_titles=row_titles,
                data=tbl_data,
            )

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
        """ """

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

    def _log_stats(self, df: pd.DataFrame):
        num_user_checkins = df["User ID"].value_counts()
        num_venue_checkins = df["Venue ID"].value_counts()
        if not self.plot_stats:
            print(f"Number of records: {df.shape[0]}")
            print(
                f"Number of users: {df['User ID'].nunique()}, with min = {num_user_checkins.min()}, max = {num_user_checkins.max()}, and avg: {num_user_checkins.mean()}"
            )
            print(
                f"Number of venues: {df['Venue ID'].nunique()}, with min = {num_venue_checkins.min()}, max = {num_venue_checkins.max()}, and avg: {num_venue_checkins.mean()}"
            )
            print(f"Number of venue categories: {df['Venue Category ID'].nunique()}")

        else:
            col_titles = ["Count", "Min Check-in", "Average Check-in", "Max Check-in"]
            row_titles = ["Users", "Venues", "Venue Ctg", "Total Records"]
            tbl_data = [
                [
                    df["User ID"].nunique(),
                    num_user_checkins.min(),
                    num_user_checkins.mean(),
                    num_user_checkins.max(),
                ],
                [
                    df["Venue ID"].nunique(),
                    num_venue_checkins.min(),
                    num_venue_checkins.mean(),
                    num_venue_checkins.max(),
                ],
                [df.shape[0], 0, 0, 0],
                [df.shape[0], 0, 0, 0]
            ]
            misc_utils.plot_plt_table(
                main_title="Records Stats",
                column_titles=col_titles,
                row_titles=row_titles,
                data=tbl_data,
            )

            # Plot distributions of user and venue check-ins
            fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=False)
            axes[0].hist(num_user_checkins, bins=100, color="blue", edgecolor="black")
            axes[0].set_title("Distribution of User Check-ins")
            axes[0].set_xlabel("Check-in Counts")
            axes[0].set_ylabel("Frequency")
            axes[0].set_xlim(left=0)
            axes[1].hist(num_venue_checkins, bins=100, color="green", edgecolor="black")
            axes[1].set_title("Distribution of Venue Check-ins")
            axes[1].set_xlabel("Check-in Counts")
            axes[1].set_ylabel("Frequency")
            axes[1].set_xlim(left=0)
            axes[1].set_yscale("log")

            plt.tight_layout()
            plt.show()

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
