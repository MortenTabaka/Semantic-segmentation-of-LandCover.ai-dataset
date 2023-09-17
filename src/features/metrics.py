import tensorflow as tf


class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    """
    Source: https://github.com/tensorflow/tensorflow/issues/32875#issuecomment-707316950
    """
    def __init__(self, num_classes=None, name=None, dtype=None):
        super(CustomMeanIoU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
