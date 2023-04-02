import os
from glob import glob
from os import path
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.data.image_preprocessing import ImagePreprocessor
from src.features.loss_functions import SemanticSegmentationLoss
from src.models.predict_model import Predictor
from src.features.data_features import ImageFeatures


class PredictionPipeline:
    OPTIMIZER = tf.keras.optimizers.Adam()

    def __init__(self, model_revision: str, input_folder: Path, output_folder: Path):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.revision_predictor = Predictor(model_revision)
        self.prediction_model = Predictor(
            model_revision
        ).get_prediction_model_of_revision
        self.model_build_parameters = self.revision_predictor.get_model_build_parameters
        self.image_features = ImageFeatures(
            self.revision_predictor.get_required_input_shape_of_an_image[0],
            self.revision_predictor.get_required_input_shape_of_an_image[1],
        )

    def process(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        InteractiveSession(config=config)

        tiles_folder = self.__preprocess_images_and_get_path(
            self.revision_predictor.get_required_input_shape_of_an_image[0]
        )
        tiles = self.__get_input_tiles(tiles_folder)
        self.__make_predictions(tiles)

        self.__clear_cache([tiles_folder])

    def __preprocess_images_and_get_path(self, targeted_tile_size: int) -> str:
        save_to = path.join(self.input_folder, ".cache/tiles")
        ImagePreprocessor(self.input_folder).split_custom_images_before_prediction(
            targeted_tile_size, save_to
        )
        return save_to

    def __make_predictions(self, tiles: List[str]):
        for tile in tiles:
            preprocessed_tile = self.get_image_for_prediction(tile)
            file_name = os.path.basename(tile)
            prediction = tf.argmax(
                self.prediction_model.predict(np.array([preprocessed_tile])), axis=-1
            )
            self.__save_prediction(prediction, file_name)

    def __save_prediction(self, predicted_mask, file_name):
        pass

    def __concatenate_tiles(self):
        pass

    def get_image_for_prediction(self, filepath: str):
        return self.image_features.load_image_from_drive(filepath)

    @staticmethod
    def __get_input_tiles(tiles_folder: str) -> List[str]:
        img_paths = glob(path.join(tiles_folder, "*.jpg"))
        return img_paths

    @staticmethod
    def __clear_cache(paths: List[str]):
        for path_to_remove in paths:
            os.removedirs(path_to_remove)
