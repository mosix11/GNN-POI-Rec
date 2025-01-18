import torch

from pytorch_lightning import LightningDataModule

from pathlib import Path

from ..utils import misc_utils



class FoursquareNYC(LightningDataModule):
    def __init__(self,
                 data_dir:Path = Path('./data').absolute(),
                 batch_size:int = 32,
                 img_size:tuple = (32, 32),
                 num_workers:int = 8,
                 num_views:int = 2,
                 exclude_classes:list = [],
                 seed:int = 11) -> None:
        
        pass

    def prepare_data(self):
        # Download or prepare data if needed
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

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
    
    
    
    def _download_dataset(self):
        zip_file_path = self.dataset_base_dir.joinpath(Path('NMR_Dataset.zip'))
        if zip_file_path.exists():
            print('Dataset already downloaded!')
        else:
            try:
                url = "https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip"
                print("Starting download...")
                misc_utils.download_file_fast(url, zip_file_path.absolute())
                
            except Exception as e:
                raise RuntimeError(f"Failed to download the dataset: {e}")
        if self.objects_base_dir.exists():
            print("Dataset already extracted!")
        else:
            try:
                print("Extracting the file...")
                self.objects_base_dir.mkdir(exist_ok=False)
                misc_utils.extract_zip_multithreaded(zip_file_path.absolute(), self.dataset_base_dir.absolute())
                print(f"Extraction completed. Files are available in: {self.objects_base_dir}")
            except Exception as e:
                shutil.rmtree(self.objects_base_dir)
                raise RuntimeError(f"Failed to extraxt the dataset: {e}")