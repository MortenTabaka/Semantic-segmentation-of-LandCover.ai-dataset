import tensorflow as tf

from src.models.architectures import (
    deeplabv3plus,
    modified_deeplabv3plus,
    modified_deeplabv3plus_v8_3,
    modified_deeplabv3plus_v8_4,
    modified_deeplabv3plus_v8_5,
)


class Model:
    def __init__(
        self,
        image_height: int,
        image_width: int,
        number_of_classes: int,
    ):
        """
        Class describing a single Tensorflow2 model.

        Args:
            image_height: height of a single image
            image_width: width of a single image
            number_of_classes: number of classes in classification
        """
        self.image_height = image_height
        self.image_width = image_width
        self.number_of_classes = number_of_classes

    def get_deeplab_model(
        self,
        weights: str = None,
        freeze_layers: bool = False,
        custom_freeze_border: int = None,
        activation: str = None,
        architecture_version: str = None,
    ) -> tf.keras.Model:
        """
        Creates model Deeplabv3plus or its modification.
        Original Tensorflow2 implementation: https://github.com/bonlime/keras-deeplab-v3-plus

        Args:
            architecture_version: one of "original", "v1", "v2", "v3", "v4"
            weights: one of 'pascal_voc' (pre-trained on pascal voc),
                'cityscapes' (pre-trained on cityscape) or None (random initialization)
            freeze_layers: should some layers be not trainable;
                must set value for border by using custom_freeze_border;
            custom_freeze_border: below that layer, all will be frozen
            activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None

        Returns:

        """

        arguments = [
            weights,
            None,
            (self.image_height, self.image_width, 3),
            self.number_of_classes,
            "xception",
            16,
            1.0,
            activation,
        ]

        if architecture_version == "v1":
            model = modified_deeplabv3plus.Deeplabv3(*arguments)
        elif architecture_version == "v2":
            model = modified_deeplabv3plus_v8_3.Deeplabv3(*arguments)
        elif architecture_version == "v3":
            model = modified_deeplabv3plus_v8_4.Deeplabv3(*arguments)
        elif architecture_version == "v4":
            model = modified_deeplabv3plus_v8_5.Deeplabv3(*arguments)
        else:
            model = deeplabv3plus.Deeplabv3(*arguments)

        if freeze_layers and custom_freeze_border:
            return self.freeze_model_layers(model, custom_freeze_border)

        return model

    @staticmethod
    def freeze_model_layers(
        model: tf.keras.models.Model, custom_freeze_border: int
    ) -> tf.keras.Model:
        for i, layer in enumerate(model.layers):
            if i < custom_freeze_border:
                layer.trainable = False
            else:
                layer.trainable = True
        return model
