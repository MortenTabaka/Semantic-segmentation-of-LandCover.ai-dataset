import tensorflow as tf


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
        image = image / 255.0
        return image


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

    def load_mask_from_drive(self, absolute_mask_path):
        """
        Read and decode a single mask.
        Args:
            absolute_mask_path: absolute path to a single image file

        """
        single_mask = tf.io.read_file(absolute_mask_path)
        single_mask = tf.image.decode_png(single_mask, channels=3)
        single_mask = single_mask[..., 0]
        single_mask = tf.reshape(single_mask, (self.image_height, self.image_width, 1))
        single_mask.set_shape([None, None, 1])
        single_mask = tf.image.resize(
            images=single_mask, size=[self.image_height, self.image_width]
        )
        return single_mask

    def encode_mask_to_one_hot(self, single_mask):
        if single_mask.shape[-1] == 1:
            single_mask = tf.one_hot(
                tf.cast(single_mask[..., 0], tf.int32), self.number_of_classes
            )
            single_mask = tf.cast(single_mask, tf.float32)
        return single_mask