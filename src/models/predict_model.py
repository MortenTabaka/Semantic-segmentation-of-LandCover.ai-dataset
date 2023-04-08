import os
from typing import Tuple

import tensorflow as tf

from src.data.requests_downloader import UrlDownloader
from src.features.model_features import (
    get_model_build_params_for_revision,
    get_revision_model_architecture,
)
from src.models.model_builder import build_deeplabv3plus


class Predictor:
    def __init__(self, model_key: str, which_metric_best_weights_to_load: str = "miou"):
        """
        Initializes a Predictor object.

        Args:
            model_key (str): Name of the revision from `models/models_revision.yaml`, e.g. `deeplabv3plus_v5.10.1`.
            which_metric_best_weights_to_load (str): Model has saved weights for best miou and loss metrics.
        """
        self.model_key = model_key
        self.url_with_zipped_weights = (
            f"https://huggingface.co/MortenTabaka/LandcoverSemanticSegmentation"
            f"/resolve/main/Weights/{model_key}.zip"
        )
        self.chosen_weights_with_best_metric = which_metric_best_weights_to_load

    def get_multiple_batches_predictions(self, multiple_batch):
        """
        Performs predictions on multiple batches of images.

        Args:
            multiple_batch (List[Tuple[np.ndarray, np.ndarray]]): A list of tuples containing input images and corresponding ground-truth masks.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of tuples containing input images and predicted masks.
        """
        model = self.get_prediction_model_of_revision
        images_and_predictions = []
        for single_batch in multiple_batch:
            images = single_batch[0]
            y_pred = tf.argmax(model.predict(images), axis=-1)
            images_and_predictions.append((images, y_pred))
        return images_and_predictions

    @property
    def get_prediction_model_of_revision(self):
        """
        Loads the prediction model for the specified revision.

        Returns:
            tf.keras.Model: A TensorFlow Keras model for semantic segmentation prediction.
        """
        model_name = get_revision_model_architecture(self.model_key)
        model = build_deeplabv3plus(model_name, self.get_model_build_parameters)
        model.load_weights(self.get_model_revision_weights).expect_partial()
        return model

    @property
    def get_model_revision_weights(self):
        """
        Downloads and loads the weights for the specified revision model.

        Returns:
            str: Path to the downloaded checkpoint file.
        """
        downloader = UrlDownloader()
        weights_path = os.path.join(downloader.get_project_root(), "data/weights")
        downloader.download_single_zip_file(
            url=self.url_with_zipped_weights,
            file_name=f"{self.model_key}.zip",
            output_path=weights_path,
            unzip=True,
        )

        return os.path.join(
            *[
                weights_path,
                self.model_key,
                f"best_{self.chosen_weights_with_best_metric}",
                "checkpoint",
            ]
        )

    @property
    def get_required_input_shape_of_an_image(self) -> Tuple[int, int, int]:
        """
        Gets the required shape of input images for the initialized Predictor.

        Returns:
            Tuple[int, int, int]: The height, width, and number of channels of the input image expected by the model.
        """
        return self.get_model_build_parameters[2]

    @property
    def get_model_build_parameters(self):
        """
        Gets the build parameters for the initialized Predictor's model.
        """
        return get_model_build_params_for_revision(self.model_key)

    @property
    def get_model_architecture(self):
        """
        Gets the architecture name of the initialized Predictor's model.

        Returns:
            str: The name of the architecture used by the model.
        """
        return get_revision_model_architecture(self.model_key)

    @staticmethod
    def get_single_batch_prediction(single_batch, model):
        """
        Performs prediction on a single batch of images.

        Args:
            single_batch (Tuple[np.ndarray, np.ndarray]): A tuple containing input images and corresponding ground-truth masks.
            model (tf.keras.Model): A TensorFlow Keras model for semantic segmentation prediction.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing input images, ground-truth masks, and predicted masks.
        """
        images = single_batch[0]
        y_pred = tf.argmax(model.predict(images), axis=-1)
        y_true = tf.argmax(single_batch[1], axis=-1)
        return images, y_true, y_pred
