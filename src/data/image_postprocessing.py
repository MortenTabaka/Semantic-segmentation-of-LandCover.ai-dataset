import glob
import os
import os.path
from pathlib import Path
from re import match
from typing import List, Union

import cv2
import numpy as np

from src.data.image_preprocessing import ImagePreprocessor


class ImagePostprocessor:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        self.input_path = input_path
        self.output_path = output_path

    def concatenate_images(self):
        img_filenames = sorted(os.listdir(self.input_path))
        img_tiles = []
        for img_filename in img_filenames:
            img_tile = cv2.imread(os.path.join(self.input_path, img_filename))
            img_tiles.append(img_tile)

        img_shape = img_tiles[0].shape
        num_cols = int(np.ceil(np.sqrt(len(img_tiles))))
        num_rows = int(np.ceil(len(img_tiles) / num_cols))
        img = np.zeros(
            (num_rows * img_shape[0], num_cols * img_shape[1], img_shape[2]),
            dtype=np.uint8,
        )

        k = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if k >= len(img_tiles):
                    break
                img[
                    i * img_shape[0] : (i + 1) * img_shape[0],
                    j * img_shape[1] : (j + 1) * img_shape[1],
                    :,
                ] = img_tiles[k]
                k += 1
        # Get the base filename
        base_filename = os.path.splitext(img_filenames[0])[0].split("_vertical")[0]
        output_filename = os.path.join(self.output_path, base_filename + ".jpg")
        cv2.imwrite(output_filename, img)

    def get_all_filepaths_of_images_in_folder(self) -> List[str]:
        """
        Returns: List of filepaths to all TIF, JPG and PNG images.
        """
        img_paths = glob.glob(os.path.join(self.input_path, "*.tiff"))
        img_paths += glob.glob(os.path.join(self.input_path, "*.jpg"))
        img_paths += glob.glob(os.path.join(self.input_path, "*.png"))
        img_paths.sort()

        return img_paths

    @staticmethod
    def __get_all_base_names_from_list_of_tiles(file_names: List[str]) -> List[str]:
        vertical = ImagePreprocessor.NAMING_CONVENTION_FOR_VERTICAL_TILE
        horizontal = ImagePreprocessor.NAMING_CONVENTION_FOR_HORIZONTAL_TILE
        second_part_of_name_for_first_tile = f"_{vertical}0_{horizontal}0"

        all_base_names = []
        for filename in file_names:
            if second_part_of_name_for_first_tile in filename:
                base_name = filename.split("_vertical0_horizontal0")[0]
                all_base_names.append(base_name)

        return all_base_names
