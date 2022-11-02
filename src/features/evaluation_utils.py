from __future__ import annotations

import json
import os
from collections import defaultdict
from json import dump

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.features.future_tensorflow import IoU


class HistoryUtilities:
    def __init__(self, model_history_filepath):
        self.filepath = model_history_filepath

    def dump_model_history_to_file(self, tensorflow_model_history):
        """
        Saves model history generated during training.
        Args:
            tensorflow_model_history: Tensorflow history; history = model.fit(...)
        """
        history = tensorflow_model_history.history
        dump(history, open(self.filepath, "w"))

    def load_model_history_from_file(self) -> dict:
        return json.load(open(self.filepath, "r"))


class History:
    def __init__(self):
        pass

    def display_history_plots(
        self,
        model_histories: list[dict] | list[str],
        save_to_folder: str,
    ):
        """
        Displays plots for train and validation values.
        Args:
            model_histories: list[dict] | list[str]: history dictionaries or its filepaths.
            save_to_folder: str: path to folder where images should be saved
        """
        if all(isinstance(item, dict) for item in model_histories):
            all_history = self.merge_multiple_histories(histories=model_histories)
        elif all(isinstance(item, str) for item in model_histories):
            all_history = self.merge_multiple_histories(
                histories_filepaths=model_histories
            )
        else:
            all_history = defaultdict()
            print("History paths or dictionaries were not passed to function.")

        if not os.path.exists(save_to_folder):
            os.makedirs(save_to_folder)

        if save_to_folder[-1] != "/":
            save_to_folder += "/"

        try:
            training_keys = []
            # validation_keys = []

            for key in all_history.keys():
                if "val" not in key:
                    print(key)
                    training_keys.append(key)

            for key in training_keys:
                validation_key = "val_" + str(key)
                number_of_epochs = len(all_history[key])

                plt.figure(figsize=(12, 10), dpi=140)
                plt.plot(np.arange(1, number_of_epochs + 1, step=1), all_history[key])
                plt.plot(
                    np.arange(1, number_of_epochs + 1, step=1),
                    all_history[validation_key],
                )
                plt.title(f"Training {key} (epoch)", fontsize="x-large")
                plt.ylabel(f"{key}".capitalize(), fontsize="large")
                plt.xlabel("Epoch", fontsize="large")
                plt.legend(
                    [f"{key}".capitalize(), f"Validation {key}"], fontsize="large"
                )
                plt.xticks(np.arange(1, number_of_epochs + 1, step=1))
                plt.grid()
                plt.savefig(save_to_folder + f"{key}.jpg")

        except AttributeError as err:
            print(
                "Validate if passed arguments are correct."
                "\nPython dictionary of model history or its filepath to JSON must be passed."
                f"\n{err}"
            )

    @staticmethod
    def merge_multiple_histories(
        histories: None | list[dict] = None,
        histories_filepaths: None | list[str] = None,
    ) -> defaultdict[list]:
        """
        Merges multiple histories into one.
        Pass only one history's dictionaries or filepaths.
        Args:
            histories: list[dict]: list of histories dictionaries
            histories_filepaths: list[str]: list of history filepaths

        Returns: Merged histories according to key.
        """
        added_histories = []

        if histories:
            for history in histories:
                added_histories.append(history)
        elif histories_filepaths:
            for filepath in histories_filepaths:
                history = HistoryUtilities(filepath).load_model_history_from_file()
                added_histories.append(history)

        merged = defaultdict(list)
        for single_history in added_histories:
            for key, value in single_history.items():
                merged[key] += value

        return merged


class ConfusionMatrix:
    def __init__(
        self,
        trained_model: tf.keras.Model,
        dataset: tf.data.Dataset,
        number_of_classes: int,
    ):
        self.model = trained_model
        self.dataset = dataset
        self.number_of_classes = number_of_classes

    def get_dataframe(self) -> pd.DataFrame:
        df_matrix = pd.DataFrame(
            0,
            index=[i for i in range(self.number_of_classes)],
            columns=[i for i in range(self.number_of_classes)],
        )

        for single_batch in self.dataset:

            y_true, y_pred = self.get_predictions_and_labels(single_batch)

            matrix = tf.math.confusion_matrix(
                labels=y_true,
                predictions=y_pred,
                num_classes=self.number_of_classes,
            ).numpy()

            batch_df = pd.DataFrame(
                matrix,
                index=[i for i in range(self.number_of_classes)],
                columns=[i for i in range(self.number_of_classes)],
            )

            df_matrix += batch_df

        return df_matrix

    def get_predictions_and_labels(self, single_batch):
        images = single_batch[0]
        y_pred = tf.argmax(self.model.predict(images), axis=-1)
        y_true = tf.argmax(single_batch[1], axis=-1)

        y_pred = tf.reshape(
            y_pred, [y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2]]
        )
        y_true = tf.reshape(
            y_true, [y_true.shape[0] * y_true.shape[1] * y_true.shape[2]]
        )

        return y_true, y_pred


class PredictionIoU:
    def __init__(
            self,
            trained_model: tf.keras.Model,
            dataset: tf.data.Dataset,
            number_of_classes: int,
    ):
        self.trained_model = trained_model
        self.dataset = dataset
        self.number_of_classes = number_of_classes

    def get_iou_for_every_class(self, save_directory=None):
        iou_per_class = []
        iou = []

        for class_id in range(self.number_of_classes):
            iou_per_class.append(IoU(self.number_of_classes, [class_id]))

        for single_batch in self.dataset:
            y_true, y_pred = ConfusionMatrix(
                self.trained_model,
                self.dataset,
                self.number_of_classes,
            ).get_predictions_and_labels(single_batch)

            for m in iou_per_class:
                m.update_state(y_true, y_pred)

        for m in iou_per_class:
            iou.append(m.result().numpy())

        df = pd.DataFrame(iou,
                          index=[i for i in range(self.number_of_classes)],
                          columns=['IoU score'])
        if save_directory:
            if save_directory[-1] != "/":
                save_directory += "/"
            save_path = save_directory + 'iou_for_every_class.csv'
            df.to_csv(save_path)
            print(f'CSV saved to {save_path}')
        else:
            print('CSV won\'t be saved')

        return df
