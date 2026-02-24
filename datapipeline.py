import torch
import os
import tarfile
import zipfile
from urllib.parse import urlparse

class DataPipeline:
    _dataset = {'PROTAC-8K': 'https://zenodo.org/records/14715718/files/PROTAC-8K.zip?download=1'}

    @classmethod
    def download_dataset(cls, dataset_name: str, directory: str):
        if dataset_name not in cls._dataset:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        os.makedirs(directory, exist_ok=True)
        dataset_url = cls._dataset[dataset_name]

        parsed_url = urlparse(dataset_url)
        file_name = os.path.basename(parsed_url.path)
        save_path = os.path.join(directory, file_name)
        if os.path.exists(save_path):
            return save_path

        response = requests.get(dataset_url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)

        return save_path


        @classmethod
        def extract_dataset(cls, save_path: str, directory: str):
            os.makedirs(directory, exist_ok= True)
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(directory)

            return directory

if __name__ == "__main__":
    zip_path = DataPipeline.download_dataset("PROTAC-8K", "zip_data")
    DataPipeline.extract_dataset(zip_path, "data/PROTAC") 