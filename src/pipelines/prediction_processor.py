import os
from glob import glob
from os import path
from pathlib import Path
from shutil import rmtree
from typing import List
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.segmentation import slic
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.data.image_postprocessing import ImagePostprocessor
from src.data.image_preprocessing import ImagePreprocessor
from src.features.data_features import ImageFeatures
from src.features.model_features import decode_segmentation_mask_to_rgb
from src.features.utils import generate_colormap
from src.models.predict_model import Predictor


class PredictionPipeline:
    """
    A class for predicting segmentation masks using a trained deep learning model.

    Parameters:
        model_revision (str): The version of the model to use for prediction.
        input_folder (Path): The folder containing the input images.
        output_folder (Path): The folder where the predicted segmentation masks will be saved.
        which_metric_best_weights_to_load (str): Weights to load.

    Attributes:
        input_folder (Path): The folder containing the input images.
        output_folder (Path): The folder where the predicted segmentation masks will be saved.
        revision_predictor (Predictor): An instance of the Predictor class that uses the specified model_revision.
        prediction_model (Model): The prediction model of the specified model_revision.
        model_build_parameters (dict): A dictionary of the model build parameters used for training the model.
        image_features (ImageFeatures): An instance of the ImageFeatures class used for loading and preprocessing images.

    Methods:
        process(): Processes the input images and saves the predicted segmentation masks to the output folder.
    """

    def __init__(
        self,
        model_revision: str,
        input_folder: Path,
        output_folder: Path,
        which_metric_best_weights_to_load: str,
        tiles_superpixel_postprocessing,
        number_of_superpixels: int = 300,
        compactness: float = 10,
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.revision_predictor = Predictor(model_revision)
        self.prediction_model = Predictor(
            model_revision, which_metric_best_weights_to_load
        ).get_prediction_model_of_revision
        self.model_build_parameters = self.revision_predictor.get_model_build_parameters
        self.image_features = ImageFeatures(
            self.revision_predictor.get_required_input_shape_of_an_image[0],
            self.revision_predictor.get_required_input_shape_of_an_image[1],
        )
        self.tiles_superpixel_postprocessing = tiles_superpixel_postprocessing
        self.params_of_superpixels_postprocessing = (number_of_superpixels, compactness)

    def process(
        self, clear_cache: bool = True, superpixels_postprocessing: bool = True
    ):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        InteractiveSession(config=config)

        tiles_folder = self.__preprocess_images_and_get_path(
            self.revision_predictor.get_required_input_shape_of_an_image[0]
        )

        tiles = self.__get_input_tiles(tiles_folder)
        self.__make_predictions(tiles)

        predicted_tiles = os.path.join(self.output_folder, ".cache/prediction_tiles")
        self.__concatenate_tiles(predicted_tiles)

        if clear_cache:
            self.__clear_cache()

    def __preprocess_images_and_get_path(self, targeted_tile_size: int) -> str:
        save_to = path.join(self.input_folder, ".cache/tiles")
        ImagePreprocessor(self.input_folder).split_custom_images_before_prediction(
            targeted_tile_size, save_to
        )
        return save_to

    def __make_predictions(
        self, tiles: List[str]
    ):
        num_classes, custom_colormap = self.__get_number_of_classes_and_colormap
        for tile in tqdm(tiles, desc='Processing tiles', unit='tile'):
            preprocessed_tile = self.__get_image_for_prediction(tile)
            file_name = os.path.basename(tile)
            prediction = tf.argmax(
                self.prediction_model.predict(np.array([preprocessed_tile])), axis=-1
            )

            if self.tiles_superpixel_postprocessing:
                segments = self.__get_superpixel_segments(tile)
                num_of_segments = self.__get_number_of_segments(segments)
                prediction = (
                    self.__get_updated_prediction_with_postprocessor_superpixels(
                        prediction, segments, num_of_segments
                    )
                )

            prediction = decode_segmentation_mask_to_rgb(
                prediction, custom_colormap, num_classes
            )
            self.__save_prediction(prediction, file_name)

    def __save_prediction(self, image, file_name):
        save_to = path.join(self.output_folder, ".cache/prediction_tiles")
        os.makedirs(save_to, exist_ok=True)
        file_path = os.path.join(save_to, file_name)
        image.save(file_path)

    def __concatenate_tiles(self, input_folder):
        ImagePostprocessor(
            input_path=input_folder, output_path=self.output_folder
        ).concatenate_images()

    def __get_image_for_prediction(self, filepath: str):
        return self.image_features.load_image_from_drive(filepath)

    def __clear_cache(self, paths=None):
        if paths is None:
            paths = [
                os.path.join(self.input_folder, ".cache"),
                os.path.join(self.output_folder, ".cache"),
            ]
        for path_to_remove in paths:
            rmtree(path_to_remove)

    def __get_superpixel_segments(
        self, image: str
    ) -> tf.Tensor:
        """
        Generates superpixel segments for an input image using the Simple Linear
        Iterative Clustering (SLIC) algorithm.

        Args:
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
        segments = slic(image, *self.params_of_superpixels_postprocessing, start_label=0)
        segments = tf.convert_to_tensor(segments)
        segments = tf.reshape(
            segments,
            (1, self.image_features.image_height, self.image_features.image_width),
        )
        return segments

    @property
    def __get_number_of_classes_and_colormap(self):
        num_classes = self.model_build_parameters[3]
        if num_classes == 5:
            custom_colormap = (
                [0, 0, 0],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 255],
            )
        else:
            custom_colormap = generate_colormap(num_classes)
        return num_classes, custom_colormap

    @staticmethod
    def __get_number_of_segments(superpixel_segments: tf.Tensor) -> int:
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
    def __get_updated_prediction_with_postprocessor_superpixels(
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

    @staticmethod
    def __get_input_tiles(tiles_folder: str) -> List[str]:
        img_paths = glob(path.join(tiles_folder, "*.jpg"))
        return img_paths
