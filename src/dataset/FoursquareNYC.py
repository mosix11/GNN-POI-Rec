import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    
    def __init__(self,
                 trajectories,
                 ):
        super().__init__()
        
        self.trajectories = trajectories
        
        
    def __len__(self):
        return self.trajectories.shape[0]
    
    def __getitem__(self, idx):
        traj = self.trajectories.iloc[idx]
        item = [
            torch.tensor(traj['User ID']),
            torch.tensor(traj['Venue ID']),
            torch.tensor(traj['Venue Category ID']),
            torch.tensor(traj['Geohash ID']),
            torch.tensor(traj['Time Slot']),
            torch.tensor(traj['Unix Timestamp'])
        ]
        return item
    
    
    @staticmethod
    def custom_collate(batch, max_seq_length: int, sampling_method: str, train:bool=True):
        users, pois, pois_cat, gh, ts, ut = zip(*batch)

        # Helper function to process each sequence based on the sampling method
        def process_sequence(seq):
            L = len(seq)
            if L > max_seq_length:
                if sampling_method == 'window':
                    # Sample a continuous interval of length max_seq_length
                    start_idx = random.randint(0, L - max_seq_length)
                    seq = seq[start_idx:start_idx + max_seq_length]
                elif sampling_method == 'random':
                    # Sample max_seq_length indices randomly
                    indices = sorted(random.sample(range(L), max_seq_length))
                    seq = [seq[i] for i in indices]
                    
            if not isinstance(seq, torch.Tensor):
                seq = torch.tensor(seq)
            return seq, len(seq)

        processed_pois = [process_sequence(seq) for seq in pois]
        processed_pois_cat = [process_sequence(seq) for seq in pois_cat]
        processed_gh = [process_sequence(seq) for seq in gh]
        processed_ts = [process_sequence(seq) for seq in ts]
        processed_ut = [process_sequence(seq) for seq in ut]

        pois, pois_lens = zip(*processed_pois)
        pois_cat, pois_cat_lens = zip(*processed_pois_cat)
        gh, gh_lens = zip(*processed_gh)
        ts, ts_lens = zip(*processed_ts)
        ut, ut_lens = zip(*processed_ut)

        # Pad sequences to max_seq_length
        pois = pad_sequence(pois, batch_first=True)
        pois_cat = pad_sequence(pois_cat, batch_first=True)
        gh = pad_sequence(gh, batch_first=True)
        ts = pad_sequence(ts, batch_first=True)
        ut = pad_sequence(ut, batch_first=True)

        users = torch.tensor(users)

        orig_lens = torch.tensor(pois_lens)

        # Prepare x and y for training with teacher forcing
        if train:
            x = (users, pois[:, :-1], pois_cat[:, :-1], gh[:, :-1], ts[:, :-1], ut[:, :-1])
            y = (users, pois[:, 1:], pois_cat[:, 1:], gh[:, 1:], ts[:, 1:], ut[:, 1:])
            return x, y, orig_lens
        else:
            # return users, pois, pois_cat, gh, ts, ut
            return users, pois
            
        


class FoursquareNYC(LightningDataModule):
    def __init__(self,
                 data_dir:Path = Path('./data').absolute(),
                 batch_size:int = 32,
                 num_workers:int = 8,
                 user_checkin_tsh:int = (20, np.inf),
                 venue_checkin_tsh:int = (10, np.inf),
                 num_test_checkins:int = 6,
                 geohash_precision:list = [6],
                 max_traj_length:list = 64,
                 traj_sampling_method:str = 'window',
                 temporal_graph_jaccard_mult_set:bool = True,
                 temporal_graph_jaccard_sim_tsh:float = 0.9,
                 spatial_graph_self_loop:bool = True,
                 temporal_graph_self_loop:bool = True,
                 seed:int = 11) -> None:
        
        super().__init__()
        

        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.FNYC_dir = self.data_dir.joinpath(Path('FNYC'))
        self.FNYC_dir.mkdir(exist_ok=True)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.DF_COLUMNS = ['User ID',
                        'Venue ID',
                        'Venue Category ID',
                        'Venue Category Name',
                        'Latitude',
                        'Longitude',
                        'Timezone Offset',
                        'UTC Time']
        
        self.num_test_checkins = num_test_checkins
        self.user_checkin_tsh = user_checkin_tsh
        self.venue_checkin_ths = venue_checkin_tsh
        
        self.geohash_precision = geohash_precision
        
        self.max_traj_length = max_traj_length
        self.traj_sampling_method = traj_sampling_method
        
        self.temporal_graph_jaccard_mult_set = temporal_graph_jaccard_mult_set
        self.temporal_graph_jaccard_sim_tsh = temporal_graph_jaccard_sim_tsh
        self.spatial_graph_self_loop = spatial_graph_self_loop
        self.temporal_graph_self_loop = temporal_graph_self_loop
        
        
        
        
    def prepare_data(self):
        # Download or prepare data if needed
        self._download_dataset()
        self._load_data()
        self._preprocess_data()
        
        self._form_spatial_graph()
        self._form_temporal_graph()
        


    def setup(self, stage: str = None):
        # Split the dataset into train/val/test and assign to self variables
        if stage == 'fit' or stage is None:
            self.train_dataset = UserTrajectoryDataset(self.user_train_trajectories)
            self.val_dataset = UserTrajectoryDataset(self.user_test_trajectories)
        if stage == 'test' or stage is None:
            self.test_dataset = UserTrajectoryDataset(self.user_test_trajectories)
        if stage == 'predict' or stage is None:
            self.predict_dataset = ...

    def train_dataloader(self):
        collate_fn = partial(UserTrajectoryDataset.custom_collate, max_seq_length=self.max_traj_length, sampling_method=self.traj_sampling_method)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        max_length = self.num_test_checkins
        collate_fn = partial(UserTrajectoryDataset.custom_collate, max_seq_length=max_length, sampling_method=self.traj_sampling_method, train=False)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):
        max_length = self.num_test_checkins
        collate_fn = partial(UserTrajectoryDataset.custom_collate, max_seq_length=max_length, sampling_method=self.traj_sampling_method, train=False)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn,pin_memory=True, num_workers=self.num_workers)

    
    
    def _form_spatial_graph(self):
        geohash_vec = np.array(self.poi_trajectories['Geohash ID'])
        
        adj_mat = (geohash_vec[:, None] == geohash_vec).astype(int)
        if not self.spatial_graph_self_loop: np.fill_diagonal(adj_mat, 0)  # Remove self-loops
        self.spatial_graph = self.convert_to_sparse_tensor(adj_mat)
        print(f"Spatial graph sparsity: {1 -self.spatial_graph.density()}")
        
    def _form_temporal_graph(self):
        num_venues = len(self.poi_trajectories['Venue ID'])
        num_timeslots = 56
        M = np.zeros((num_venues, num_timeslots), dtype=int)
        
        for i, row in enumerate(self.poi_trajectories['Time Slot']):
            for t in row:
                M[i, t] += 1
                
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
            intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0
        )

        # Step 3: Threshold the similarity to construct the adjacency matrix
        adj_mat = (jaccard_similarity > self.temporal_graph_jaccard_sim_tsh).astype(int)
        if not self.temporal_graph_self_loop:
            np.fill_diagonal(adj_mat, 0)  # Remove self-loops

        # Step 4: Convert to sparse tensor and store
        self.temporal_graph = self.convert_to_sparse_tensor(adj_mat)
        print(f'Temporal graph sparsity: {1 - self.temporal_graph.density()}')

    
    def _preprocess_data(self):
        
        self.df = self.df.drop_duplicates()
        
        print('Dateset statistics before filtering:')
        self._log_stats(self.df)
        
        df_flt = self._filter_user_venue(self.df.copy(deep=True))
        df_flt = self._filter_user_venue(df_flt)
        print('Dateset statistics after filtering:')
        self._log_stats(df_flt)
        df_flt = self._reassign_IDs(df_flt)
        
        df_flt = self._process_time(df_flt)
        df_flt = self._process_location(df_flt)
        
        # user_trajectories, poi_trajectories = self._from_trajectories(df_flt)
        user_train_trajectories, user_test_trajectories, poi_trajectories = self._from_trajectories_with_split(df_flt, self.num_test_checkins)
        
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

    def _from_trajectories_with_split(self, df, num_test_checkins):
        # Initialize lists for train and test trajectories
        user_train_trajectories = []
        user_test_trajectories = []

        # Store training data for calculating POI trajectories later
        train_checkins = []

        for user_id, user_data in df.groupby("User ID", sort=False):
            user_data = user_data.sort_values(by="Unix Timestamp")  # Ensure temporal ordering
            train_data = user_data.iloc[:-num_test_checkins] if len(user_data) > num_test_checkins else user_data.iloc[0:0]
            test_data = user_data.iloc[-num_test_checkins:] if len(user_data) >= num_test_checkins else user_data

            if not train_data.empty:
                train_checkins.append(train_data)  # Collect for POI trajectory calculation
                user_train_trajectories.append(
                    {
                        "User ID": user_id,
                        "Venue ID": train_data["Venue ID"].tolist(),
                        "Venue Category ID": train_data["Venue Category ID"].tolist(),
                        "Geohash ID": train_data["Geohash ID"].tolist(),
                        "Local Time": train_data["Local Time"].tolist(),
                        "Time Slot": train_data["Time Slot"].tolist(),
                        "Unix Timestamp": train_data["Unix Timestamp"].tolist(),
                    }
                )

            if not test_data.empty:
                user_test_trajectories.append(
                    {
                        "User ID": user_id,
                        "Venue ID": test_data["Venue ID"].tolist(),
                        "Venue Category ID": test_data["Venue Category ID"].tolist(),
                        "Geohash ID": test_data["Geohash ID"].tolist(),
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
        poi_trajectories = train_data_only.groupby("Venue ID", sort=False).agg(
            {
                "Geohash": "first",
                "Geohash ID": "first",
                "Venue Category ID": "first",
                "User ID": list,
                "Local Time": list,
                "Time Slot": list,
                "Unix Timestamp": list,
            }
        ).reset_index()
        poi_trajectories = poi_trajectories.sort_values(by=["Venue ID"])

        return user_train_trajectories_df, user_test_trajectories_df, poi_trajectories  
    # def _from_trajectories(sefl, df):
    #     user_trajectories = df.groupby("User ID", sort=False).agg(
    #         {
    #             "Venue ID": list,
    #             "Venue Category ID": list,
    #             "Geohash ID":list,
    #             "Local Time": list,
    #             "Time Slot":list,
    #             "Unix Timestamp": list,
    #             # 'Normalized Timestamp': list
    #         }
    #     ).reset_index()
        
        
    #     poi_trajectories = df.groupby("Venue ID", sort=False).agg(
    #         {
    #             "Geohash": "first",
    #             "Geohash ID": "first",
    #             "Venue Category ID": "first",
    #             "User ID": list,
    #             "Local Time": list,
    #             "Time Slot": list,
    #             "Unix Timestamp": list,
    #             # 'Normalized Timestamp': list
    #         }
    #     ).reset_index()
    #     poi_trajectories = poi_trajectories.sort_values(by=['Venue ID'])
        
    #     return user_trajectories, poi_trajectories
    
    def _process_location(self, df):
        # for precision in self.geohash_precision:
        #     df[f"geohash@{precision}"] = df.apply(lambda row: geohash.encode(row['Latitude'], row['Longitude'], precision=precision), axis=1)
        df['Geohash'] = df.apply(lambda row: geohash.encode(row['Latitude'], row['Longitude'], precision=self.geohash_precision[0]), axis=1)
        df = df.drop(columns=['Latitude', 'Longitude'])
        
        # Create a mapping dictionary for geohash to integer ID
        unique_geohashes = df['Geohash'].unique()
        geohash_to_id = {geohash: idx for idx, geohash in enumerate(unique_geohashes, start=1)}
        id_to_geohash = {idx: geohash for geohash, idx in geohash_to_id.items()}  # Reverse map for recovery
        # Convert geohash to integer IDs
        df['Geohash ID'] = df['Geohash'].map(geohash_to_id)

        # df = df.drop(columns=['Geohash'])
        # Store the mapping for recovery later (optional, depending on use case)
        self.GEOHASH_TO_ID_MAP = geohash_to_id
        self.ID_TO_GEOHASH_MAP = id_to_geohash
        return df
    
    def _process_time(self, df):
        
        def calculate_local_time(utc_time, timezone_offset):
            # Parse the UTC time string
            utc_datetime = datetime.strptime(utc_time, '%a %b %d %H:%M:%S +0000 %Y')
            # Add the timezone offset
            local_datetime = utc_datetime + timedelta(minutes=timezone_offset)
            return local_datetime
        
        def local_time_to_week_slot(df) -> int:
            # Extract the day of the week (Monday=0, Sunday=6) and hour
            df['Day of Week'] = df['Local Time'].dt.weekday
            df['Hour'] = df['Local Time'].dt.hour

            # Calculate the time slot (each day has 8 slots, each slot is 3 hours)
            # Slot formula: (Day of Week * 8) + (Hour // 3)
            df['Time Slot'] = (df['Day of Week'] * 8) + (df['Hour'] // 3)

            # Drop intermediate columns if not needed
            df = df.drop(columns=['Day of Week', 'Hour'])

            return df
        
        df['Local Time'] = df.apply(lambda row: calculate_local_time(row['UTC Time'], row['Timezone Offset']), axis=1)
        df['Unix Timestamp'] = df['Local Time'].apply(lambda x: int(x.timestamp()))
        df = local_time_to_week_slot(df)
        # min_timestamp = df['Unix Timestamp'].min()
        # max_timestamp = df['Unix Timestamp'].max()
        # df['Normalized Timestamp'] = (df['Unix Timestamp'] - min_timestamp) / (max_timestamp - min_timestamp)
        df = df.sort_values(by=['User ID', 'Unix Timestamp'])
        df = df.drop(columns=['Timezone Offset', 'UTC Time'])
        
        return df
    
    def _reassign_IDs(self, df):
        df['User ID'] = pd.factorize(df['User ID'])[0] + 1
        venue_id_mapping = {id_: idx for idx, id_ in enumerate(df['Venue ID'].unique(), start=1)}
        venue_category_id_mapping = {id_: idx for idx, id_ in enumerate(df['Venue Category ID'].unique(), start=1)}
        df['Venue ID'] = df['Venue ID'].map(venue_id_mapping)
        df['Venue Category ID'] = df['Venue Category ID'].map(venue_category_id_mapping)
        venue_category_id_name_mapping = df[['Venue Category ID', 'Venue Category Name']].drop_duplicates()
        venue_category_id_name_mapping = dict(zip(venue_category_id_name_mapping['Venue Category ID'], 
                                                venue_category_id_name_mapping['Venue Category Name']))

        df = df.drop(columns=['Venue Category Name'])
        self.VENUE_CAT_ID_MAP = venue_category_id_name_mapping
        return df
        

    def _log_stats(self, df):
        num_user_checkins = df['User ID'].value_counts()
        num_venue_checkins = df['Venue ID'].value_counts()
        print(f"Number of users: {df['User ID'].nunique()}, with min = {num_user_checkins.min()}, max = {num_user_checkins.max()}, and avg: {num_user_checkins.mean()}")
        print(f"Number of venues: {df['Venue ID'].nunique()}, with min = {num_venue_checkins.min()}, max = {num_venue_checkins.max()}, and avg: {num_venue_checkins.mean()}")
        print(f"Number of venue categories: {df['Venue Category ID'].nunique()}")
        
    def _filter_user_venue(self, df):
        num_user_checkins = df['User ID'].value_counts()
        num_venue_checkins = df['Venue ID'].value_counts()
        # num_venue_users = self.df.groupby("Venue ID")["User ID"].nunique()
        # venues_with_enough_users = num_venue_users[(num_venue_users >= self.venue_checkin_ths[0]) & (num_venue_users <= self.venue_checkin_ths[1])].index
        venues_with_enough_users = num_venue_checkins[(num_venue_checkins >= self.venue_checkin_ths[0]) & (num_venue_checkins <= self.venue_checkin_ths[1])].index
        df = df[df['Venue ID'].isin(venues_with_enough_users)]
        
        num_user_checkins = df['User ID'].value_counts()
        num_venue_checkins = df['Venue ID'].value_counts()
        
        filter_idxs = num_user_checkins[(num_user_checkins >= self.user_checkin_tsh[0]) & (num_user_checkins <=self.user_checkin_tsh[1])].index
        df = df[df['User ID'].isin(filter_idxs)]
        df = df.sort_values(by=['User ID'])
        return df
        
    
    def _load_data(self):
        df = pd.read_csv(self.raw_file_path,  sep='\t', encoding='latin-1', names=self.DF_COLUMNS)
        self.df = df.sort_values(by=['User ID'])
        # column_null_counts = df.isnull().sum()  # Count of 'null' values in each column
        # print(f"Number of rows: {df.shape[0]} | Number of columns: {df.shape[1]} ")
        # print("\nSummary of 'null' values in each column:")
        # print(column_null_counts)
    
    def _download_dataset(self):
        zip_file_path = self.FNYC_dir.joinpath(Path('dataset.zip'))
        if zip_file_path.exists():
            print('Dataset already downloaded!')
        else:
            try:
                url = "http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
                print("Starting download...")
                misc_utils.download_file_fast(url, zip_file_path.absolute())
                
            except Exception as e:
                raise RuntimeError(f"Failed to download the dataset: {e}")
            
        self.raw_files_dir = self.FNYC_dir.joinpath('raw_files')
        if self.raw_files_dir.exists():
            print("Dataset already extracted!")
        else:
            try:
                print("Extracting the file...")
                self.raw_files_dir.mkdir(exist_ok=False)
                misc_utils.extract_zip_multithreaded(zip_file_path.absolute(), self.raw_files_dir.absolute())
                print(f"Extraction completed. Files are available in: {self.raw_files_dir}")
            except Exception as e:
                shutil.rmtree(self.raw_files_dir)
                raise RuntimeError(f"Failed to extraxt the dataset: {e}")
            
        self.raw_file_path = self.raw_files_dir.joinpath('dataset_tsmc2014/dataset_TSMC2014_NYC.txt')
        
        
        
    def search_user_venue_visits(self, df, user_id, venue_id):
        user_row = df[df['User ID'] == user_id]
        
        if user_row.empty:
            return f"User ID {user_id} not found in the DataFrame."
        user_row = user_row.iloc[0]
        
        visit_indices = [i for i, v in enumerate(user_row['Venue ID']) if v == venue_id]
        
        if not visit_indices:
            return f"Venue ID {venue_id} not visited by User ID {user_id}."
        
        visit_times = [user_row['Local Time'][i] for i in visit_indices]
        
        result = pd.DataFrame({
            "Venue ID": [venue_id] * len(visit_indices),
            "Local Time": visit_times
        })
        
        return result
    
    
    def get_sparsity(self, mat):
        total_elements = mat.size
        num_zeroes = np.sum(mat == 0)
        percentage_zeroes = (num_zeroes / total_elements) * 100
        return percentage_zeroes
    
    def convert_to_sparse_tensor(sefl, mat):
        # Convert the matrix to a sparse tensor
        row, col = np.where(mat)  # Find indices of non-zero elements
        sparse_tensor = SparseTensor(row=torch.tensor(row), col=torch.tensor(col), value=torch.tensor(mat[row, col]))
        return sparse_tensor
    
    
    def plot_distribution(self, data, x_label='Value', y_label='Frequency'):
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins='auto', edgecolor='black', alpha=0.7)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('Distribution Plot')
        plt.show()