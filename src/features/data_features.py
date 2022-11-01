import tensorflow as tf
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input


class ImageFeatures:
    def __init__(
        self,
        image_height: int,
        image_width: int,
    ):
        self.image_height = image_height
        self.image_width = image_width

    def load_image_from_drive(self, absolute_image_path):
        """
        Read, decode and normalize a single image.
        Args:
            absolute_image_path: absolute path to a single image file

        """
        image = tf.io.read_file(absolute_image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            images=image, size=[self.image_height, self.image_width]
        )
        image = self.normalize_image(image)
        return image

    @staticmethod
    def normalize_image(image):
        return preprocess_input(image, mode="tf")


class MaskFeatures:
    def __init__(
        self,
        mask_height: int,
        mask_width: int,
        number_of_classes: int,
    ):
        self.image_height = mask_height
        self.image_width = mask_width
        self.number_of_classes = number_of_classes

    def load_mask_from_drive(self, absolute_mask_path: str) -> tf.image:
        """
        Read and decode a single mask.
        Args:
            absolute_mask_path: absolute path to a single image file

        """
        single_mask = tf.io.read_file(absolute_mask_path)
        single_mask = tf.image.decode_png(single_mask, channels=1)
        single_mask = tf.image.resize(
            images=single_mask, size=[self.image_height, self.image_width]
        )
        single_mask = tf.cast(single_mask, tf.int32)
        single_mask = tf.one_hot(single_mask[..., 0], self.number_of_classes)
        return single_mask
