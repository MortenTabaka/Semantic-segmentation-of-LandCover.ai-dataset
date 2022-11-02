import tensorflow as tf


class Predictor:
    def __init__(self, trained_model: tf.keras.Model):
        self.model = trained_model

    def get_single_batch_prediction(self, single_batch):
        images = single_batch[0]
        y_pred = tf.argmax(self.model.predict(images), axis=-1)
        y_true = tf.argmax(single_batch[1], axis=-1)

        y_pred = tf.reshape(
            y_pred, [y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2]]
        )
        y_true = tf.reshape(
            y_true, [y_true.shape[0] * y_true.shape[1] * y_true.shape[2]]
        )

        return images, y_true, y_pred
