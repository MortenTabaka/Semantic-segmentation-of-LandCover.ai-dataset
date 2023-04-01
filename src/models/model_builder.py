import tensorflow as tf
from typing import List, Union

from src.features.utils import revision_a_model
from src.models.architectures import (
    deeplabv3plus,
    modified_v1_deeplabv3plus,
    modified_v2_deeplabv3plus,
    modified_v3_deeplabv3plus,
    modified_v4_deeplabv3plus,
)


class Model:
    def __init__(
        self,
        revision: str,
        batch_size: int,
        input_image_height: int,
        input_image_width: int,
        number_of_classes: int,
        pretrained_weights: str = None,
        do_freeze_layers: bool = False,
        last_layer_frozen: int = None,
        activation: str = None,
        model_architecture: str = None,
        output_stride: int = 16,
        optimizer: tf.keras.optimizers.Optimizer = None,
        loss_function: tf.keras.losses.Loss = None,
        metrics: Union[
            tf.keras.metrics.Metric, str, List[Union[tf.keras.metrics.Metric, str]]
        ] = None,
    ):
        """
        Class describing a single Tensorflow2 model.
        Original Tensorflow2 implementation: https://github.com/bonlime/keras-deeplab-v3-plu

        Args:
            input_image_height: height of a single image
            input_image_width: width of a single image
            number_of_classes: number of classes in classification
            pretrained_weights: one of 'pascal_voc' (pre-trained on pascal voc),
                'cityscapes' (pre-trained on cityscape) or None (random initialization)
            do_freeze_layers: should some layers be not trainable;
                must set value for border by using last_layer_frozen;
            last_layer_frozen: below that layer, all will be frozen
            activation: optional activation to add to the top of the network.
                One of 'softmax', 'sigmoid' or None
            model_architecture: one of "original", "v1", "v2", "v3", "v4"
            output_stride: determines input_shape/feature_extractor_output ratio. One of {8,16}.
        """
        self.model_build_parameters = [
            pretrained_weights,
            None,
            (input_image_height, input_image_width, 3),
            number_of_classes,
            "xception",
            output_stride,
            1.0,
            activation,
        ]
        self.revision = revision
        self.batch_size = batch_size
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width
        self.number_of_classes = number_of_classes
        self.pretrained_weights = pretrained_weights
        self.do_freeze_layers = do_freeze_layers
        self.last_layer_frozen = last_layer_frozen
        self.activation = activation
        self.model_architecture = model_architecture
        self.output_stride = output_stride
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def get_deeplab_model(self) -> tf.keras.Model:
        """
        Build a Tensorflow2 model.
        """
        if self.output_stride not in (8, 16):
            print("output_stride must be 8 or 16. output_stride will be set to 16.")
            self.output_stride = 16

        if self.model_architecture == "original":
            model = deeplabv3plus.Deeplabv3(*self.model_build_parameters)
        elif self.model_architecture == "v1":
            model = modified_v1_deeplabv3plus.Deeplabv3(*self.model_build_parameters)
        elif self.model_architecture == "v2":
            model = modified_v2_deeplabv3plus.Deeplabv3(*self.model_build_parameters)
        elif self.model_architecture == "v3":
            model = modified_v3_deeplabv3plus.Deeplabv3(*self.model_build_parameters)
        elif self.model_architecture == "v4":
            model = modified_v4_deeplabv3plus.Deeplabv3(*self.model_build_parameters)
        else:
            model = deeplabv3plus.Deeplabv3(*self.model_build_parameters)

        if self.do_freeze_layers and self.last_layer_frozen:
            return self.freeze_model_layers(model, self.last_layer_frozen)

        return model

    def save_model_revision(self):
        revision_a_model(
            self.get_deeplab_model().name,
            self.revision,
            self.batch_size,
            self.input_image_height,
            self.input_image_width,
            self.number_of_classes,
            self.pretrained_weights,
            self.do_freeze_layers,
            self.last_layer_frozen,
            self.activation,
            self.model_architecture,
            self.output_stride,
        )

    @property
    def get_compile_parameters(self):
        return [self.optimizer, self.loss_function, self.metrics]

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
