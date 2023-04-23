import os
from glob import glob
from os import path
from pathlib import Path
from shutil import rmtree
from typing import List, Union
from tqdm import tqdm

import numpy as np
from skimage.io import imread
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.data.image_postprocessing import ImagePostprocessor, SuperpixelsProcessor
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
        image_features (ImageFeatures): An instance of the ImageFeatures class used for loading and preprocessing images

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
        self.raw_image_height = (
            self.revision_predictor.get_required_input_shape_of_an_image[0]
        )
        self.raw_image_width = (
            self.revision_predictor.get_required_input_shape_of_an_image[1]
        )
        self.image_features = ImageFeatures(
            self.raw_image_height,
            self.raw_image_width,
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
            for image in self.__get_full_size_images:
                loaded_image = imread(image)
                SuperpixelsProcessor(loaded_image, self.get_slic_parameters)

        if clear_cache:
            self.__clear_cache()

    def __preprocess_images_and_get_path(self, targeted_tile_size: int) -> str:
        save_to = path.join(self.input_folder, ".cache/tiles")
        ImagePreprocessor(self.input_folder).split_custom_images_before_prediction(
            targeted_tile_size, save_to
        )
        return save_to

    def __make_predictions(self, tiles: List[str]):
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
                prediction, *self.__get_number_of_classes_and_colormap
            )
            self.__save_prediction(decoded_prediction, file_name)

    def __get_superpixel_post_processed_tile_prediction(
        self, tile: str, prediction: tf.Tensor
    ) -> tf.Tensor:
        image = imread(tile)
        prediction = SuperpixelsProcessor(
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
        ).concatenate_all_tiles()

    def __get_image_for_prediction(self, filepath: str):
        return self.image_features.load_image_from_drive(filepath)

    def postprocess_tiles_borders_in_concatenated_prediction(
        self,
        raw_image,
        decoded_image,
        tile_height: int,
        tile_width: int,
        output: Union[str, Path],
        border_pixel_range: int = 100,
    ):
        vertical_borders = int(self.raw_image_height / tile_height) - 1
        horizontal_borders = self.raw_image_width / tile_width - 1

        # TODO: Repeat for horizontal
        for vertical_num in range(vertical_borders):
            top = (vertical_num + 1) * tile_height + border_pixel_range
            bottom = (vertical_num + 1) * tile_height - border_pixel_range
            raw_border_area = raw_image[bottom:top, :]
            decoded_border_image = decoded_image[bottom:top, :]
            # TODO: Encode or decode or save not decoded(may be the best)
            not_decoded = decoded_border_image
            decoded_border_image = (
                SuperpixelsProcessor(raw_border_area, self.get_slic_parameters).
                get_updated_prediction_with_postprocessor_superpixels(
                    not_decoded, 0.3
                    )
            )
            decoded_image[bottom:top, :] = decoded_border_image

        decoded_image.save(self.output_folder)

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
        """
        Source: https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
        Params:
            n_segments – The (approximate) number of labels in the segmented output image.
            compactness – Balances color proximity and space proximity.
                Higher values give more weight to space proximity, making superpixel shapes more square/cubic.
                In SLICO mode, this is the initial compactness. This parameter depends strongly on image contrast and
                on the shapes of objects in the image. We recommend exploring possible values on a log scale,
                e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.
            max_iter – Maximum number of iterations of k-means.
            sigma – Width of Gaussian smoothing kernel for pre-processing for each dimension of the image.
                The same sigma is applied to each dimension in case of a scalar value. Zero means no smoothing.
                Note, that sigma is automatically scaled if it is scalar and a manual voxel spacing is provided
                (see Notes section).
            spacing – The voxel spacing along each image dimension. By default, slic assumes uniform spacing
                (same voxel resolution along z, y and x).
                This parameter controls the weights of the distances along z, y, and x during k-means clustering.
            multichannel – Whether the last axis of the image is to be interpreted as multiple channels
                or another spatial dimension.
            convert2lab – Whether the input should be converted to Lab colorspace prior to segmentation.
                The input image must be RGB. Highly recommended. This option defaults to True
                when multichannel=True and image.shape[-1] == 3.
            enforce_connectivity – Whether the generated segments are connected or not
            min_size_factor – Proportion of the minimum segment size to be removed with respect to the supposed segment
                size `depth*width*height/n_segments`
            max_size_factor – Proportion of the maximum connected segment size. A value of 3 works in most of the cases.
            slic_zero – Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
            start_label – The labels' index start. Should be 0 or 1.
            mask – If provided, superpixels are computed only where mask is True, and seed points are
                homogeneously distributed over the mask using a K-means clustering strategy

                Returns: dict

        """
        return {
            "n_segments": self.number_of_superpixels,
            "compactness": self.compactness,
            "max_iter": 10,
            "sigma": 0,
            "spacing": None,
            "multichannel": True,
            "convert2lab": None,
            "enforce_connectivity": True,
            "min_size_factor": 0.5,
            "max_size_factor": 3,
            "slic_zero": False,
            "start_label": 0,
            "mask": None,
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
        return custom_colormap, num_classes

    @property
    def __get_full_size_images(self) -> List[str]:
        return glob(path.join(self.output_folder, "*.jpg"))

    @staticmethod
    def __get_input_tiles(tiles_folder: str) -> List[str]:
        img_paths = glob(path.join(tiles_folder, "*.jpg"))
        return img_paths
