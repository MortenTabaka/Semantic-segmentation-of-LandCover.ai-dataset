from typing import List, Tuple
from pathlib import Path

import numpy as np

import tensorflow as tf

from src.models.predict_model import Predictor


class PredictionPipeline:
    def __init__(
        self,
        model_revision: str,
        input_folder: Path,
        output_folder: Path
    ):
        self.predictor = Predictor(model_revision)
        self.model = self.predictor.get_prediction_model_of_revision
        self.image_input_specs = self.predictor.get_required_input_shape_of_an_image
        self.input_folder = input_folder
        self.output_folder = output_folder

    def processing_pipeline(self):
        pass

    def load_images(self):
        pass

    def make_predictions(self):
        pass

    def concatenate_tiles(self):
        pass

    def save_predictions(self):
        pass

    def make_prediction_for_single_tile(self, single_image: np.array):
        if tf.shape(single_image) == ():
            return tf.argmax(self.model.predict(single_image), axis=-1)
        else:
            raise ValueError("Input image has incompatible shape.")