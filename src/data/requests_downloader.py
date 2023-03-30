import os
import shutil
import zipfile
from os import makedirs, path

import requests
from tqdm import tqdm


class UrlDownloader:
    def __init__(self):
        pass

    def download_single_zip_file(
        self, url: str, file_name: str, output_path: str, unzip: bool = True
    ):
        write_to = path.join(output_path, file_name)
        if not path.isfile(write_to):
            if not path.exists(output_path):
                makedirs(output_path)
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            with open(write_to, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            print(f"File saved to:\n{write_to}")
            if unzip:
                print("Unzipping.")
                self.unzip_file(write_to, output_path)
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR: Download incomplete")

        elif unzip and path.isfile(write_to):
            if path.exists(output_path):
                print("Previously unzipped files exists.")
            else:
                print("Unzipping already existing file.")
                self.unzip_file(write_to, output_path)
        else:
            print("File already exists.")

    def download_project_preprocessed_dataset(self):
        file_name: str = "preprocessed_images.zip"
        path_from_root: str = "data"
        url = "https://huggingface.co/datasets/MortenTabaka/LandCover-Aerial-Imagery-for-semantic-segmentation/resolve/main/landcover_processed_for_training.zip"
        output = os.path.join(self.get_project_root(), path_from_root)
        self.download_single_zip_file(url, file_name, output, True)

    @staticmethod
    def unzip_file(zip_path: str, extract_path: str):
        # Open the zip file for reading
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract all files to the specified directory
            zip_ref.extractall(extract_path)

    @staticmethod
    def get_project_root() -> str:
        """Returns the absolute path of the project root directory."""
        current_dir = os.path.abspath(os.curdir)
        while not os.path.isfile(os.path.join(current_dir, "README.md")):
            current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        return current_dir
