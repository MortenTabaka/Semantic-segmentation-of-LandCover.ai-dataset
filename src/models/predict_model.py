import os

import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file

from src.models.model_builder import build_deeplabv3plus
from src.features.utils import (
    get_model_build_params_for_revision,
    get_revision_model_architecture,
)


class Predictor:
    def __init__(self, model_key: str):
        """
        Args:
            model_key: name of revision from models/models_revision.yaml, e.g. deeplabv3plus_v5.10.1
        """
        self.model_key = model_key
        self.url_with_zipped_weights = (
            f"https://huggingface.co/MortenTabaka/LandcoverSemanticSegmentation"
            f"/resolve/main/Weights/{model_key}.zip"
        )

    def get_multiple_batches_predictions(self, multiple_batch):
        model = self.build_prediction_model_of_revision()
        model.load_weights(self.get_model_revision_weights())
        images_and_predictions = []
        for single_batch in multiple_batch:
            images = single_batch[0]
            y_pred = tf.argmax(model.predict(images), axis=-1)
            images_and_predictions.append((images, y_pred))
        return images_and_predictions

    def get_single_batch_prediction(self, single_batch):
        model = self.build_prediction_model_of_revision()
        model.load_weights(self.get_model_revision_weights())
        images = single_batch[0]
        y_pred = tf.argmax(model.predict(images), axis=-1)
        y_true = tf.argmax(single_batch[1], axis=-1)

        return images, y_true, y_pred

    def build_prediction_model_of_revision(self):
        build_params = get_model_build_params_for_revision(self.model_key)
        model_name = get_revision_model_architecture(self.model_key)
        return build_deeplabv3plus(model_name, build_params)

    def get_model_revision_weights(self):
        """
        Download and load revision model weights.
        Returns:
            model: tf.keras.Model
        """

        weights = get_file(
            fname=f"{self.model_key}.zip",
            origin=self.url_with_zipped_weights,
            extract=True,
            cache_subdir="weights",
        )
        return weights
