from pathlib import Path

import numpy as np
from os.path import join as join_paths

import tensorflow as tf

from src.models.predict_model import Predictor
from src.features.loss_functions import SemanticSegmentationLoss
from src.data.image_preprocessing import ImagePreprocessor


class PredictionPipeline:
    OPTIMIZER = tf.keras.optimizers.Adam()

    def __init__(self, model_revision: str, input_folder: Path, output_folder: Path):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.revision_predictor = Predictor(model_revision)
        self.prediction_model = Predictor(model_revision).get_prediction_model_of_revision
        self.model_build_parameters = self.revision_predictor.get_model_build_parameters

    def process(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        InteractiveSession(config=config)

        tiles_folder = self.__preprocess_images_and_get_path(
            self.revision_predictor.get_required_input_shape_of_an_image[0]
        )
        tiles = self.__get_input_tiles(tiles_folder)

        for tile in tiles:
            preprocessed_tile = img_to_array(load_img(tile))
            prediction = tf.argmax(
                self.prediction_model.predict(preprocessed_tile), axis=-1
            )
            print(type(prediction))

        self.__clear_cache([tiles_folder])

    def __preprocess_images_and_get_path(self, targeted_tile_size: int) -> str:
        save_to = path.join(self.input_folder, ".cache/tiles")
        ImagePreprocessor(self.input_folder).split_custom_images_before_prediction(
            targeted_tile_size, save_to
        )
        return save_to

    def make_predictions(self):
        pass

    def concatenate_tiles(self):
        pass

    def save_predictions(self):
        pass

    @staticmethod
    def __get_input_tiles(tiles_folder: str) -> List[str]:
        img_paths = glob(path.join(tiles_folder, "*.jpg"))
        return img_paths

    @staticmethod
    def __clear_cache(paths: List[str]):
        for path_to_remove in paths:
            os.removedirs(path_to_remove)
