from __future__ import annotations

import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import matplotlib.pyplot as plt

from src.features.dataset import Dataset
from src.models.predict_model import Predictor


COLORMAP = ([0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255])


class PredictionMasks:
    def __init__(
        self,
        trained_model: tf.keras.Model,
        dataset_object: Dataset,
        number_of_classes: int,
    ):
        self.model = trained_model
        self.dataset = dataset_object
        self.num_classes = number_of_classes

    # def infer(model, image_tensor):
    #     predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    #     predictions = np.squeeze(predictions)
    #     predictions = np.argmax(predictions, axis=2)
    #     return predictions

    def display_overlay_predictions_on_test_set(self, colormap=COLORMAP):
        _, _, test_dataset = self.dataset.generate_datasets()
        predictor = Predictor(self.model)
        for single_batch in test_dataset:
            images, y_true, y_pred = predictor.get_single_batch_prediction(single_batch)

            for image, mask_true, mask_pred in zip(images, y_true, y_pred):
                mask_true = self.decode_segmentation_mask(
                    mask_true, colormap, self.num_classes
                )
                mask_pred = self.decode_segmentation_mask(
                    mask_pred, colormap, self.num_classes
                )
                overlay = self.get_overlay(image, mask_pred)
                overlay_original = self.get_overlay(image, mask_true)
                self.plot_samples_matplotlib(
                    [image, overlay_original, overlay, mask_pred],
                    figure_size=(18, 14),
                )

    @staticmethod
    def decode_segmentation_mask(
        landcover_mask, custom_colormap: list[list[float]], num_classes: int
    ):
        """
        Transforms Landcover dataset's masks to RGB image.

        Args:
            landcover_mask: prediction or true mask;
            custom_colormap: user-defined colormap; len(custom_colormap) == num_classes;
                E.g. [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
                    [[R, G, B], [R, G, B], ...]
            num_classes: number of classes;
        """
        if len(custom_colormap) != num_classes:
            raise AttributeError("")
        r = np.zeros_like(landcover_mask).astype(np.uint8)
        g = np.zeros_like(landcover_mask).astype(np.uint8)
        b = np.zeros_like(landcover_mask).astype(np.uint8)
        for i in range(0, num_classes):
            idx = landcover_mask == i
            r[idx] = custom_colormap[i][0]
            g[idx] = custom_colormap[i][1]
            b[idx] = custom_colormap[i][2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    @staticmethod
    def get_overlay(image, colored_mask):
        image = tf.keras.preprocessing.image.array_to_img(image)
        image = np.array(image).astype(np.uint8)
        image = tf.image.resize(image, [512, 512])
        image.set_shape([None, None, 3])
        image = tf.reshape(image, (512, 512, 3))
        overlay = tfa.image.blend(image, colored_mask, 0.5)
        return overlay

    @staticmethod
    def plot_samples_matplotlib(display_list, figure_size=(5, 3)):
        _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figure_size)
        for i in range(len(display_list)):
            if display_list[i].shape[-1] == 3:
                axes[i].imshow(
                    tf.keras.preprocessing.image.array_to_img(display_list[i])
                )
            else:
                axes[i].imshow(display_list[i])
        plt.show()
