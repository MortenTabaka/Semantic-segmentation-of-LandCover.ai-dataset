import os
from glob import glob
from os import path
from pathlib import Path
from shutil import rmtree
from typing import List, Union

import numpy as np
import tensorflow as tf
from cv2 import imread, imwrite
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tqdm import tqdm

from src.data.image_postprocessing import (
    ImagePostprocessor,
    SuperpixelsProcessor,
    DataMode,
)
from src.data.image_preprocessing import ImagePreprocessor
from src.features.data_features import ImageFeatures
from src.features.model_features import (
    decode_segmentation_mask_to_rgb,
)
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
        self.tile_height = self.revision_predictor.get_required_input_shape_of_an_image[
            0
        ]
        self.tile_width = self.revision_predictor.get_required_input_shape_of_an_image[
            1
        ]
        self.image_features = ImageFeatures(
            self.tile_height,
            self.tile_width,
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
            ImagePostprocessor(
                os.path.join(self.output_folder, ".cache/logits_prediction_tensors"),
                self.output_folder,
                DataMode.NUMPY_TENSOR,
            ).concatenate_all_tiles()

            for raw_image, raw_mask in zip(
                self.__get_full_size_raw_images,
                sorted(glob(path.join(self.output_folder, "*.npy"))),
            ):
                raw_image = imread(raw_image)
                raw_mask = np.load(raw_mask)
                self.__postprocess_tiles_borders_in_concatenated_prediction(
                    raw_image, raw_mask
                )

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

            self.__save_prediction_tensor_in_numpy_file(prediction, file_name)

            decoded_prediction = decode_segmentation_mask_to_rgb(
                prediction, *self.__get_colormap_and_number_of_classes
            )
            self.__save_prediction_as_decoded_image(decoded_prediction, file_name)

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

    def __save_prediction_as_decoded_image(self, image, file_name):
        save_to = path.join(self.output_folder, ".cache/prediction_tiles")
        os.makedirs(save_to, exist_ok=True)
        file_path = os.path.join(save_to, file_name)
        image.save(file_path)

    def __save_prediction_tensor_in_numpy_file(
        self, prediction: tf.Tensor, file_name: str
    ):
        save_to = path.join(self.output_folder, ".cache/logits_prediction_tensors")
        os.makedirs(save_to, exist_ok=True)
        file_name = file_name.replace(".jpg", ".npy")
        file_path = os.path.join(save_to, file_name)
        np.save(file_path, prediction.numpy())

    def __concatenate_tiles(self, input_folder):
        ImagePostprocessor(
            input_path=input_folder, output_path=self.output_folder
        ).concatenate_all_tiles()

    def __get_image_for_prediction(self, filepath: str):
        return self.image_features.load_image_from_drive(filepath)

    def __postprocess_tiles_borders_in_concatenated_prediction(
        self,
        raw_image,
        raw_mask,
        border_pixel_range: int = 50,
    ):

        width = tf.shape(raw_mask)[1]
        height = tf.shape(raw_mask)[0]

        # get exact shape as prediction since it may be smaller
        raw_image = raw_image[:height, :width, :]
        num_vertical_borders = int(width / self.tile_height) - 1
        num_horizontal_borders = int(height / self.tile_width) - 1

        raw_mask = self.__process_single_oriented_borders(
            raw_image,
            raw_mask,
            "vertical",
            num_vertical_borders,
            border_pixel_range,
        )

        raw_mask = self.__process_single_oriented_borders(
            raw_image,
            raw_mask,
            "horizontal",
            num_horizontal_borders,
            border_pixel_range,
        )

        # TODO: Remove creating this test file
        filename = os.path.join(self.output_folder, "test.jpg")
        imwrite(filename, raw_mask)

    def __process_single_oriented_borders(
        self,
        raw_image,
        raw_prediction: tf.Tensor,
        orientation: str,
        num_borders: int,
        border_pixel_range: int,
    ):

        for border in range(num_borders):
            top_or_right = (border + 1) * self.tile_height + border_pixel_range
            bottom_or_left = (border + 1) * self.tile_height - border_pixel_range

            raw_border_area, decoded_border_mask = self.__get_border_area(
                raw_image, raw_prediction, orientation, top_or_right, bottom_or_left
            )

            post_processed_border = SuperpixelsProcessor(
                raw_border_area, self.get_slic_parameters
            ).get_updated_prediction_with_postprocessor_superpixels(raw_prediction, 0.3)
            post_processed_border = decode_segmentation_mask_to_rgb(
                post_processed_border, *self.__get_colormap_and_number_of_classes
            )
            if orientation == "horizontal":
                raw_prediction[bottom_or_left:top_or_right, :] = post_processed_border
            elif orientation == "vertical":
                raw_prediction[:, bottom_or_left:top_or_right] = post_processed_border
            else:
                raise ValueError("Pick correct which border to process.")
        return raw_prediction

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
    def __get_colormap_and_number_of_classes(self):
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
    def __get_full_size_raw_images(self) -> List[str]:
        raw_images = glob(path.join(self.input_folder, "*.jpg"))
        raw_images += glob(path.join(self.input_folder, "*.png"))
        raw_images += glob(path.join(self.input_folder, "*.tif"))
        raw_images += glob(path.join(self.input_folder, "*.tiff"))
        return sorted(raw_images)

    @staticmethod
    def __get_input_tiles(tiles_folder: str) -> List[str]:
        img_paths = glob(path.join(tiles_folder, "*.jpg"))
        return img_paths

    @staticmethod
    def __get_border_area(
        raw_image,
        decoded_mask,
        orientation: str,
        top_or_right: int,
        bottom_or_left: int,
    ):
        if orientation == "horizontal":
            return (
                raw_image[bottom_or_left:top_or_right, :],
                decoded_mask[bottom_or_left:top_or_right, :],
            )
        elif orientation == "vertical":
            return (
                raw_image[:, bottom_or_left:top_or_right],
                decoded_mask[:, bottom_or_left:top_or_right],
            )
        else:
            raise ValueError("Pick correct which border to process.")
