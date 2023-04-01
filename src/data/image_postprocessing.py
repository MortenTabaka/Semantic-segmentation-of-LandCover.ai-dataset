import glob
import os
import os.path
from typing import List

import cv2


class ImagePostprocessor:
    def __init__(self, path_to_images_and_predictions: str):
        self.input_path = path_to_images_and_predictions

    def concat_image(self):
        pass

    def get_all_filepaths_of_images_in_folder(self) -> List[str]:
        """
        Returns: List of filepaths to all TIF, JPG and PNG images.
        """
        img_paths = glob.glob(os.path.join(self.input_path, "*.tif"))
        img_paths += glob.glob(os.path.join(self.input_path, "*.jpg"))
        img_paths += glob.glob(os.path.join(self.input_path, "*.png"))
        img_paths.sort()

        return img_paths
