import torch
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule

from pathlib import Path

from ..utils import misc_utils
import shutil
from datetime import datetime, timedelta



class FoursquareNYC(LightningDataModule):
    def __init__(self,
                 data_dir:Path = Path('./data').absolute(),
                 batch_size:int = 32,
                 num_workers:int = 8,
                 user_checkin_tsh:int = (20, np.inf),
                 venue_checkin_tsh:int = (10, np.inf),
                 seed:int = 11) -> None:
        
        super().__init__()
        

        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.FNYC_dir = self.data_dir.joinpath(Path('FNYC'))
        self.FNYC_dir.mkdir(exist_ok=True)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.columns = ['User ID',
                        'Venue ID',
                        'Venue Category ID',
                        'Venue Category Name',
                        'Latitude',
                        'Longitude',
                        'Timezone',
                        'UTC time']
        
        self.user_checkin_tsh = user_checkin_tsh
        self.venue_checkin_ths = venue_checkin_tsh
        
        
        
        
    def prepare_data(self):
        # Download or prepare data if needed
        self._download_dataset()
        self._load_data()
        self._preprocess_data()
        pass

    def setup(self, stage: str = None):
        # Split the dataset into train/val/test and assign to self variables
        if stage == 'fit' or stage is None:
            self.train_dataset = ...
            self.val_dataset = ...
        if stage == 'test' or stage is None:
            self.test_dataset = ...
        if stage == 'predict' or stage is None:
            self.predict_dataset = ...

    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.predict_dataset, batch_size=self.batch_size)
    
    
    def _preprocess_data(self):
        

        print('Dateset statistics before filtering:')
        print(self.df.head())
        self._log_stats(self.df)
        df_flt = self._filter_user_venue(self.df.copy(deep=True))
        df_flt = self._filter_user_venue(df_flt)
        print('Dateset statistics after filtering:')
        self._log_stats(df_flt)
        df_flt = self._reassign_IDs(df_flt)
        print(df_flt.head())
        
    
    def _process_time(self, df):
        
        def calculate_local_time(utc_time, timezone_offset):
            # Parse the UTC time string
            utc_datetime = datetime.strptime(utc_time, '%a %b %d %H:%M:%S +0000 %Y')
            # Add the timezone offset
            local_datetime = utc_datetime + timedelta(minutes=timezone_offset)
            return local_datetime
        
        df['Local_time'] = df.apply(lambda row: calculate_local_time(row['UTC_time'], row['Timezone_offset']), axis=1)
    
    def _reassign_IDs(self, df):
        venue_id_mapping = {id_: idx for idx, id_ in enumerate(df['Venue ID'].unique())}
        venue_category_id_mapping = {id_: idx for idx, id_ in enumerate(df['Venue Category ID'].unique())}
        df['Venue ID'] = df['Venue ID'].map(venue_id_mapping)
        df['Venue Category ID'] = df['Venue Category ID'].map(venue_category_id_mapping)
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
        df = pd.read_csv(self.raw_file_path,  sep='\t', encoding='latin-1', names=self.columns)
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