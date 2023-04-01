import os

import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from yaml import safe_load

from src.data.requests_downloader import UrlDownloader
from src.models.model_builder import Model
from src.features.utils import get_absolute_path_to_project_location


class Predictor:
    def __init__(self, model_key: str):
        """
        Args:
            model_key: name of revision from models/models_revision.yaml, e.g. deeplabv3plus_v5.10.1
        """
        self.model_key = model_key
        self.url_with_zipped_weights = f"https://huggingface.co/MortenTabaka/LandcoverSemanticSegmentation/resolve/main/Weights/{model_key}.zip"

    def get_single_batch_prediction(self, single_batch):
        model = Model(*self.get_model_build_parameters()).get_deeplab_model()
        images = single_batch[0]
        y_pred = tf.argmax(model.predict(images), axis=-1)
        y_true = tf.argmax(single_batch[1], axis=-1)

        return images, y_true, y_pred

    def get_model_build_parameters(self):
        yaml_filepath = get_absolute_path_to_project_location(
            "models/models_revisions.yaml"
        )
        params = []

        if os.path.exists(yaml_filepath):
            with open(yaml_filepath, "r") as f:
                existing_models_revisions = safe_load(f)
            build_params = existing_models_revisions.get(self.model_key, {}).get(
                "model_build_parameters", {}
            )
            for key in build_params:
                params.append(build_params[key])

        return params
