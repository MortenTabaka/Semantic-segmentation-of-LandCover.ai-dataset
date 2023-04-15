import glob
import os
import os.path
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.segmentation import slic

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

    @staticmethod
    def get_superpixel_segments(
        image: str, params_of_superpixels_postprocessing
    ) -> tf.Tensor:
        """
        Generates superpixel segments for an input image using the Simple Linear
        Iterative Clustering (SLIC) algorithm.

        Args:
            params_of_superpixels_postprocessing:
            image (str): File path or URL of the input image.
            num_superpixels (int, optional): Number of desired superpixels.
                Defaults to 300.
            compactness (int, optional): Compactness parameter for the SLIC
                algorithm. A higher value results in more compact and square-shaped
                superpixels, while a lower value results in more irregularly shaped
                superpixels. Defaults to 20.

        Returns:
            tf.Tensor: A tensor representing the superpixel segments, with shape
            (1, H, W), where H is the image height and W is the image width. The
            tensor contains integer values representing the segment labels (superpixel
            indices) assigned to each pixel in the image.
        """
        image = imread(image)
        segments = slic(image, *params_of_superpixels_postprocessing, start_label=0)
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

    @staticmethod
    def get_updated_prediction_with_postprocessor_superpixels(
        not_decoded_predicted_tile: tf.Tensor,
        superpixel_segments: tf.Tensor,
        num_of_segments: int,
    ):
        """
        Update the prediction for each superpixel segment in a not-decoded predicted tile.

        Args:
            not_decoded_predicted_tile (tf.Tensor): A tensor representing the not-decoded
                predicted tile, with shape (1, H, W), where H is the tile height and W is
                the tile width. The tensor contains integer values representing the predicted
                classes for each pixel in the tile.
            superpixel_segments (tf.Tensor): A tensor representing the superpixel segments,
                with shape (1, H, W), where H is the tile height and W is the tile width.
                The tensor contains integer values representing the segment labels (superpixel
                indices) assigned to each pixel in the tile.
            num_of_segments (int): The number of segments in the superpixel segmentation map.

        Returns:
            tf.Tensor: A tensor representing the updated not-decoded predicted tile, with
            shape (1, H, W), where H is the tile height and W is the tile width. The tensor
            contains integer values representing the updated predicted classes for each pixel
            in the tile, after considering the most frequent class within each superpixel segment.
        """
        for segment_num in range(num_of_segments):
            # get indices of single segment
            indices = tf.where(tf.equal(superpixel_segments, segment_num)).numpy()
            # extract the same part from prediction
            tile_extracted_part = tf.gather_nd(not_decoded_predicted_tile, indices)
            tile_extracted_part = tf.cast(tile_extracted_part, dtype=tf.int32)
            # count number of classes occurrences in extracted prediction
            counts = tf.math.bincount(tile_extracted_part)
            # Find the index of the most often repeated value
            most_frequent_value_index = tf.math.argmax(counts)
            # Get the most often repeated value
            most_frequent_class_in_tile_segment = most_frequent_value_index.numpy()

            # Create a tensor of ones with the shape of indices
            ones = tf.ones((tf.shape(indices)[0],), dtype=tf.int64)
            # Multiply the ones tensor by max_value
            updates = ones * most_frequent_class_in_tile_segment
            # Update the not_decoded_prediction tensor
            not_decoded_predicted_tile = tf.tensor_scatter_nd_update(
                not_decoded_predicted_tile, indices, updates
            )

        return not_decoded_predicted_tile
