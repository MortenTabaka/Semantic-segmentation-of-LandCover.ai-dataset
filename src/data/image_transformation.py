import tensorflow as tf
import numpy as np


class ImageTransformator:

    def __init__(
            self,
            image_height: int,
            image_width: int,
            probability_threshold: float = 0,
            min_crop_coefficient: float = 0.7,
            max_crop_coefficient: float = 1.0,
            hue_coefficient: float = 0.2,
    ):
        """
        Creates object for images and its masks transformations.
        Args:
            image_height: default height
            image_width: default width
            probability_threshold: if each transformation will take place is decided by probability;
                by default it is set to 0, so every transformation will occur.
        """
        self.image_height = image_height
        self.image_width = image_width
        self.threshold = probability_threshold
        self.min_crop_coefficient = min_crop_coefficient
        self.max_crop_coefficient = max_crop_coefficient
        self.hue_coefficient = hue_coefficient

    def get_randomly_transformed_image_and_mask(
            self,
            input_image,
            input_mask,
    ):
        # random crop and resize
        if tf.random.uniform(()) > self.threshold:
            crop_size_height = np.random.randint(
                self.min_crop_coefficient * self.image_height,
                self.max_crop_coefficient * self.image_height,
                dtype=int,
            )

            crop_size_width = np.random.randint(
                self.min_crop_coefficient * self.image_width,
                self.max_crop_coefficient * self.image_width,
                dtype=int,
            )

            input_image = tf.image.random_crop(input_image, size=(crop_size_height, crop_size_width, 3))
            input_image = tf.image.resize(input_image, [self.image_height, self.image_width])

            input_mask = tf.image.resize(input_mask, [self.image_height, self.image_width])
            input_mask = tf.image.random_crop(input_mask, size=(crop_size_height, crop_size_width, 1))
            input_mask = tf.image.resize(input_mask, [self.image_height, self.image_width])

        # random horizontal flip
        if tf.random.uniform(()) > self.threshold:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        # random vertical flip
        if tf.random.uniform(()) > self.threshold:
            input_image = tf.image.flip_up_down(input_image)
            input_mask = tf.image.flip_up_down(input_mask)

        return input_image, input_mask
