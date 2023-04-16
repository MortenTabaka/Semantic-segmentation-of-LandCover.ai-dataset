import os
from glob import glob
from os import path
from pathlib import Path
from shutil import rmtree
from typing import List
from tqdm import tqdm

import numpy as np
from skimage.io import imread
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.data.image_postprocessing import ImagePostprocessor, SlicSuperPixels
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
        tiles_superpixel_postprocessing: bool,
        number_of_superpixels: int = None,
        compactness: float = None,
        superpixel_threshold: float = None,
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
        self.number_of_superpixels = number_of_superpixels
        self.compactness = compactness
        self.superpixel_threshold = superpixel_threshold

    def process(self, postprocess_boundaries: bool, clear_cache: bool = True):
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

        if postprocess_boundaries:
            self.__post_process_tiles_boundaries_in_image()

        if clear_cache:
            self.__clear_cache()

    def __preprocess_images_and_get_path(self, targeted_tile_size: int) -> str:
        save_to = path.join(self.input_folder, ".cache/tiles")
        ImagePreprocessor(self.input_folder).split_custom_images_before_prediction(
            targeted_tile_size, save_to
        )
        return save_to

    def __make_predictions(self, tiles: List[str]):
        num_classes, custom_colormap = self.__get_number_of_classes_and_colormap
        for tile in tqdm(tiles, desc="Processing tiles", unit="tile"):
            preprocessed_tile = self.__get_image_for_prediction(tile)
            file_name = os.path.basename(tile)

            prediction = tf.argmax(
                self.prediction_model.predict(np.array([preprocessed_tile])), axis=-1
            )

            if self.tiles_superpixel_postprocessing:
                prediction = self.__get_superpixel_post_processed_tile_prediction(
                    tile, prediction
                )

            decoded_prediction = decode_segmentation_mask_to_rgb(
                prediction, custom_colormap, num_classes
            )
            self.__save_prediction(decoded_prediction, file_name)

    def __get_superpixel_post_processed_tile_prediction(
        self, tile: str, prediction: tf.Tensor
    ) -> tf.Tensor:
        image = imread(tile)
        prediction = SlicSuperPixels(
            image, self.get_slic_parameters
        ).get_updated_prediction_with_postprocessor_superpixels(
            prediction, self.superpixel_threshold
        )
        return prediction

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

    def __post_process_tiles_boundaries_in_image(self):
        """
        Post-process resulting map to improve boundaries between concatenated tiles.
        Returns:

        """
        pass

    def __clear_cache(self, paths=None):
        if paths is None:
            paths = [
                os.path.join(self.input_folder, ".cache"),
                os.path.join(self.output_folder, ".cache"),
            ]
        for path_to_remove in paths:
            rmtree(path_to_remove)

    @property
    def get_slic_parameters(self):
        return {
            'n_segments': self.number_of_superpixels,
            'compactness': self.compactness,
            'max_iter': 10,
            'sigma': 0,
            'spacing': None,
            'multichannel': True,
            'convert2lab': None,
            'enforce_connectivity': True,
            'min_size_factor': 0.5,
            'max_size_factor': 3,
            'slic_zero': False,
            'start_label': 0,
            'mask': None
        }

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
    def __get_input_tiles(tiles_folder: str) -> List[str]:
        img_paths = glob(path.join(tiles_folder, "*.jpg"))
        return img_paths
