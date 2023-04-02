import os
from typing import Tuple

import tensorflow as tf

from src.models.model_builder import build_deeplabv3plus
from src.features.model_features import (
    get_model_build_params_for_revision,
    get_revision_model_architecture,
)
from src.data.requests_downloader import UrlDownloader


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
        model = self.get_prediction_model_of_revision
        images_and_predictions = []
        for single_batch in multiple_batch:
            images = single_batch[0]
            y_pred = tf.argmax(model.predict(images), axis=-1)
            images_and_predictions.append((images, y_pred))
        return images_and_predictions

    @property
    def get_prediction_model_of_revision(self):
        model_name = get_revision_model_architecture(self.model_key)
        model = build_deeplabv3plus(model_name, self.get_model_build_parameters)
        model.load_weights(self.get_model_revision_weights)
        return model

    @property
    def get_model_revision_weights(self):
        """
        Download and load revision model weights.
        Returns:
            model: tf.keras.Model
        """
        downloader = UrlDownloader()
        weights_path = os.path.join(downloader.get_project_root(), "data/weights")
        downloader.download_single_zip_file(
            url=self.url_with_zipped_weights,
            file_name=f"{self.model_key}.zip",
            output_path=weights_path,
            unzip=True,
        )

        return os.path.join(*[weights_path, self.model_key, "checkpoint"])

    @property
    def get_required_input_shape_of_an_image(self) -> Tuple[int, int, int]:
        """
        Gets required shape of input image for the initialized Predictor.

        Returns: (image_height, image_width, channels)
        """
        data = self.get_model_build_parameters
        input_shape = (
            int(data["input_shape"]["input_image_height"]),
            int(data["input_shape"]["input_image_width"]),
            int(data["input_shape"]["channels"]),
        )
        return input_shape

    @property
    def get_model_build_parameters(self):
        return get_model_build_params_for_revision(self.model_key)

    @staticmethod
    def get_single_batch_prediction(single_batch, model):
        images = single_batch[0]
        y_pred = tf.argmax(model.predict(images), axis=-1)
        y_true = tf.argmax(single_batch[1], axis=-1)
        return images, y_true, y_pred
