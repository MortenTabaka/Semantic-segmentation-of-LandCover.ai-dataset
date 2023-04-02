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
        self.predictor = Predictor(model_revision)
        self.model = self.predictor.get_prediction_model_of_revision.compile(
            self.OPTIMIZER,
            SemanticSegmentationLoss(
                self.predictor.get_model_build_parameters[3]
            ).soft_dice_loss,
        )
        self.input_image_shape = self.predictor.get_required_input_shape_of_an_image
        self.input_folder = input_folder
        self.output_folder = output_folder

    def process(self):
        self.__preprocesses_images()

    def __preprocesses_images(self):
        ImagePreprocessor(self.input_folder).split_custom_images_before_prediction(
            self.input_image_shape[0],
            join_paths(self.input_folder, ".cache/tiles")
        )

    def __load_input_images(self):
        pass

    def make_predictions(self):
        pass

    def concatenate_tiles(self):
        pass

    def save_predictions(self):
        pass

    def make_prediction_for_single_tile(self, single_image: np.array):
        if tf.shape(single_image) == self.input_image_shape:
            return tf.argmax(self.model.predict(single_image), axis=-1)
        else:
            raise ValueError("Input image has incompatible shape.")
