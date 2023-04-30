import glob
import os
import os.path
from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import tensorflow as tf
from skimage.segmentation import slic
from tqdm import tqdm

from src.data.image_preprocessing import ImagePreprocessor
from src.features.dataset import get_normalized_class_balance_of_the_landcover_dataset


class DataMode(Enum):
    IMAGE = [".jpg", ".png", ".tiff", ".tif"]
    NUMPY_TENSOR = [".npy"]


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

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        data_mode: DataMode = DataMode.IMAGE,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.data_mode = data_mode

        self.__vertical = ImagePreprocessor.NAMING_CONVENTION_FOR_VERTICAL_TILE
        self.__horizontal = ImagePreprocessor.NAMING_CONVENTION_FOR_HORIZONTAL_TILE

    def concatenate_all_tiles(self):
        img_filenames = sorted(os.listdir(self.input_path))
        base_names = self.__get_all_base_names_from_list_of_tiles(img_filenames)
        print(base_names)
        separated_tiles_according_to_base_name = (
            self.__get_tiles_according_its_base_name(base_names, img_filenames)
        )

        for base_name, filenames in zip(
            base_names, separated_tiles_according_to_base_name
        ):
            cv2.imwrite(*self.get_concatenated_filename_and_image(base_name, filenames))

    def get_concatenated_filename_and_image(
        self, base_name: str, tiles_filenames: List[str]
    ) -> np.array:
        tiles_filenames.sort()
        img_tile = cv2.imread(os.path.join(self.input_path, tiles_filenames[0]))
        img_shape = img_tile.shape

        (
            vertical_multiplicative,
            horizontal_multiplicative,
        ) = self.get_count_of_vertical_and_horizontal_tiles(tiles_filenames)

        full_sized_tensor = np.zeros(
            (
                vertical_multiplicative * img_shape[0],
                horizontal_multiplicative * img_shape[1],
                img_shape[2],
            ),
            dtype=np.uint8,
        )

        num_of_tiles = len(tiles_filenames)

        k = 0
        for v in tqdm(
            range(vertical_multiplicative),
            desc="Concatenating tiles",
            unit="vertical",
        ):
            for h in range(horizontal_multiplicative):
                if k >= num_of_tiles:
                    break

                img_tile = self.read_from_disk(
                    os.path.join(
                        self.input_path,
                        f"{base_name}_{self.__vertical}{v}_{self.__horizontal}{h}.jpg",
                    )
                )
                full_sized_tensor[
                    v * img_shape[0] : (v + 1) * img_shape[0],
                    h * img_shape[1] : (h + 1) * img_shape[1],
                    :,
                ] = img_tile
                k += 1
        output_filename = os.path.join(self.output_path, base_name + ".jpg")
        return output_filename, full_sized_tensor

    def get_all_filepaths_of_images_in_folder(self) -> List[str]:
        """
        Returns: List of filepaths to all acceptable formats.
        It may be either an image or a numpy file.
        """
        if self.data_mode == DataMode.IMAGE:
            tiles = glob.glob(os.path.join(self.input_path, "*.tif"))
            tiles += glob.glob(os.path.join(self.input_path, "*.tiff"))
            tiles += glob.glob(os.path.join(self.input_path, "*.jpg"))
            tiles += glob.glob(os.path.join(self.input_path, "*.png"))
        elif self.data_mode == DataMode.NUMPY_TENSOR:
            tiles = glob.glob(os.path.join(self.input_path, "*.npy"))
        else:
            raise ValueError(f"Invalid input file type.")

        tiles.sort()
        return tiles

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

    def get_count_of_vertical_and_horizontal_tiles(self, tiles: List[str]):
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

    def read_from_disk(self, filename_absolute: Union[str, Path]):
        if self.data_mode == DataMode.IMAGE:
            return cv2.imread(filename_absolute)
        elif self.data_mode == DataMode.NUMPY_TENSOR:
            return np.load(filename_absolute)
        else:
            raise ValueError("Not supported data mode.")


class SuperpixelsProcessor:
    def __init__(self, raw_image: tf.Tensor, slic_parameters):
        self.raw_image = raw_image
        self.params_of_superpixels_postprocessing = slic_parameters

    def get_updated_prediction_with_postprocessor_superpixels(
        self,
        not_decoded_prediction: tf.Tensor,
        threshold: float,
        should_class_balance: bool = False,
    ):
        """
        Update the prediction for each superpixel segment in a not-decoded predicted tile.

        Args:
            not_decoded_prediction (tf.Tensor): A tensor representing the not-decoded
                predicted tile, with shape (1, H, W), where H is the tile height and W is
                the tile width. The tensor contains integer values representing the predicted
                classes for each pixel in the tile.
            threshold (float): The threshold value for updating the prediction within each superpixel
                segment. If the ratio of the most frequent predicted class count to the total number
                of pixels in the segment is equal to or greater than the threshold, the prediction for
                that segment is updated to the most frequent class.
            should_class_balance (bool): Decide if class balance should be used

        Returns:
            tf.Tensor: A tensor representing the updated not-decoded predicted tile, with
            shape (1, H, W), where H is the tile height and W is the tile width. The tensor
            contains integer values representing the updated predicted classes for each pixel
            in the tile, after considering the most frequent class within each superpixel segment.
        """
        superpixel_segments = self.get_superpixel_segments()
        num_of_segments = self.get_number_of_segments(superpixel_segments)
        class_balance = get_normalized_class_balance_of_the_landcover_dataset()

        for segment_num in range(num_of_segments):
            # get indices of single segment
            indices = tf.where(tf.equal(superpixel_segments, segment_num)).numpy()
            # extract the same part from prediction
            tile_extracted_part = tf.gather_nd(not_decoded_prediction, indices)
            tile_extracted_part = tf.cast(tile_extracted_part, dtype=tf.int32)
            # count number of classes occurrences in extracted prediction
            counts = tf.math.bincount(tile_extracted_part)

            if should_class_balance:
                counts = tf.cast(counts, dtype=tf.float32)
                num_classes = len(counts)
                counts = counts / class_balance[num_classes - 1]

            # Find the index of the most often repeated value
            most_frequent_value_index = tf.math.argmax(counts)

            # Get ratio
            number_of_all_pixels_in_segment = tf.reduce_sum(counts)
            most_frequent_count = counts[most_frequent_value_index].numpy()
            ratio = most_frequent_count / number_of_all_pixels_in_segment

            road_class_pixel_count = counts[-1]

            if ratio >= threshold:
                # Get the most often repeated value
                most_frequent_class_in_tile_segment = most_frequent_value_index.numpy()
                # Create a tensor of ones with the shape of indices
                ones = tf.ones((tf.shape(indices)[0],), dtype=tf.uint8)
                # Multiply the ones tensor by max_value
                updates = ones * most_frequent_class_in_tile_segment
                # Update the not_decoded_prediction tensor
                not_decoded_prediction = tf.tensor_scatter_nd_update(
                    not_decoded_prediction, indices, updates
                )
        return not_decoded_prediction

    def get_superpixel_segments(self) -> tf.Tensor:
        """
        Generates superpixel segments for an input image using the Simple Linear
        Iterative Clustering (SLIC) algorithm.

        Returns:
            tf.Tensor: A tensor representing the superpixel segments, with shape
            (1, H, W), where H is the image height and W is the image width. The
            tensor contains integer values representing the segment labels (superpixel
            indices) assigned to each pixel in the image.
        """
        # print(f"self.raw_image = {self.raw_image}")
        # print(f"type = {type(self.raw_image)}")
        segments = slic(
            self.raw_image,
            **self.params_of_superpixels_postprocessing,
        )
        segments = tf.convert_to_tensor(segments)
        segments = tf.reshape(
            segments,
            (1, segments.shape[0], segments.shape[1]),
        )
        return segments

    @staticmethod
    def get_number_of_segments(superpixel_segments: tf.Tensor) -> int:
        """
        Calculates the number of segments in a superpixel segmentation map.

        Args:
            superpixel_segments (tf.Tensor): A tensor representing the superpixel
                segments, with shape (1, H, W), where H is the image height and W is
                the image width. The tensor contains integer values representing the
                segment labels (superpixel indices) assigned to each pixel in the image.

        Returns:
            tf.Tensor: A tensor representing the number of segments in the superpixel
            segmentation map.
        """
        max_value = tf.reduce_max(superpixel_segments, keepdims=False, axis=(1, 2))
        max_value = max_value.numpy()[0]
        return max_value + 1
