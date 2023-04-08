import glob
import os
import os.path
from typing import List, Union
from pathlib import Path
from re import match

import cv2
import numpy as np


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
        img_paths = glob.glob(os.path.join(self.input_path, "*.tif"))
        img_paths += glob.glob(os.path.join(self.input_path, "*.jpg"))
        img_paths += glob.glob(os.path.join(self.input_path, "*.png"))
        img_paths.sort()

        return img_paths
