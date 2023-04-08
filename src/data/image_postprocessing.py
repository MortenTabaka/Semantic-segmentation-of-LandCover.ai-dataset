import glob
import os
import os.path
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

from src.data.image_preprocessing import ImagePreprocessor


class ImagePostprocessor:
    """
    Class for images postprocessing.

    Args:
        input_path: Union[str, Path]: Path to the directory containing the input image tiles.
        output_path: Union[str, Path]: Path to the directory where the concatenated images will be saved.

    Attributes:
        input_path (str): Path to the directory containing the input image tiles.
        output_path (str): Path to the directory where the concatenated images will be saved.

    Methods:
        concatenate_images(self):
            Concatenates the image tiles in the input directory and saves the resulting images in the output directory.
            Raises an error if no image tiles are found in the input directory.

        get_all_filepaths_of_images_in_folder(self) -> List[str]:
            Returns a list of filepaths to all TIF, JPG and PNG images in the input directory.

    Private Methods:
        __get_all_base_names_from_list_of_tiles(self, file_names: List[str]) -> List[str]:
            Given a list of image tile filenames, returns a list of base filenames (without the tile index).
            Example: ['image_0_0.jpg', 'image_0_1.jpg', 'image_1_0.jpg', 'image_1_1.jpg'] -> ['image']

        __get_tiles_according_its_base_name(self, base_names: List[str], tiles: List[str]) -> List[List[str]]:
            Given a list of base filenames and a list of image tile filenames, returns a list of lists of image tile filenames
            grouped by base filename.
            Example: ['image'] and ['image_0_0.jpg', 'image_0_1.jpg', 'image_1_0.jpg', 'image_1_1.jpg'] ->
                     [['image_0_0.jpg', 'image_0_1.jpg', 'image_1_0.jpg', 'image_1_1.jpg']]

        __get_count_of_vertical_and_horizontal_tiles(self, tiles: List[str]) -> Tuple[int, int]:
            Given a list of image tile filenames, returns a tuple with the number of vertical and horizontal tiles.
            Example: ['image_0_0.jpg', 'image_0_1.jpg', 'image_1_0.jpg', 'image_1_1.jpg'] -> (2, 2)
    """

    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        self.input_path = input_path
        self.output_path = output_path

        self.__vertical = ImagePreprocessor.NAMING_CONVENTION_FOR_VERTICAL_TILE
        self.__horizontal = ImagePreprocessor.NAMING_CONVENTION_FOR_HORIZONTAL_TILE

    def concatenate_images(self):
        img_filenames = sorted(os.listdir(self.input_path))
        base_names = self.__get_all_base_names_from_list_of_tiles(img_filenames)
        separated_tiles_according_to_base_name = (
            self.__get_tiles_according_its_base_name(base_names, img_filenames)
        )

        for base_name, filenames in zip(
            base_names, separated_tiles_according_to_base_name
        ):
            filenames.sort()

            img_tile = cv2.imread(os.path.join(self.input_path, filenames[0]))
            img_shape = img_tile.shape

            (
                vertical_multiplicative,
                horizontal_multiplicative,
            ) = self.__get_count_of_vertical_and_horizontal_tiles(filenames)

            img = np.zeros(
                (
                    vertical_multiplicative * img_shape[0],
                    horizontal_multiplicative * img_shape[1],
                    img_shape[2],
                ),
                dtype=np.uint8,
            )

            num_of_tiles = len(filenames)

            k = 0
            for v in range(vertical_multiplicative):
                for h in range(horizontal_multiplicative):
                    if k >= num_of_tiles:
                        break

                    img_tile = cv2.imread(
                        os.path.join(
                            self.input_path,
                            f"{base_name}_{self.__vertical}{v}_{self.__horizontal}{h}.jpg",
                        )
                    )
                    img[
                        v * img_shape[0] : (v + 1) * img_shape[0],
                        h * img_shape[1] : (h + 1) * img_shape[1],
                        :,
                    ] = img_tile
                    k += 1
            output_filename = os.path.join(self.output_path, base_name + ".jpg")
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

    def __get_all_base_names_from_list_of_tiles(
        self, file_names: List[str]
    ) -> List[str]:
        all_base_names = []
        second_part_of_name_for_first_tile = f"_{self.__vertical}0_{self.__horizontal}0"

        for filename in file_names:
            if second_part_of_name_for_first_tile in filename:
                base_name = filename.split(second_part_of_name_for_first_tile)[0]
                all_base_names.append(base_name)

        return all_base_names

    def __get_tiles_according_its_base_name(
        self, base_names: List[str], tiles: List[str]
    ) -> List[List[str]]:

        split_tiles_according_to_base_name = []

        for base_name in base_names:
            tiles_of_base_image = []

            for tile in tiles:
                if tile.split(f"_{self.__vertical}")[0] == base_name:
                    tiles_of_base_image.append(tile)

            split_tiles_according_to_base_name.append(tiles_of_base_image)

        return split_tiles_according_to_base_name

    def __get_count_of_vertical_and_horizontal_tiles(self, tiles: List[str]):
        vertical = []
        horizontal = []

        for tile in tiles:
            num_vertical_tiles = int(
                tile.split(self.__vertical)[1].split(f"_{self.__horizontal}")[0]
            )
            num_horizontal_tiles = int(tile.split(self.__horizontal)[1].split(f".")[0])

            vertical.append(num_vertical_tiles)
            horizontal.append(num_horizontal_tiles)

        return max(vertical) + 1, max(horizontal) + 1
