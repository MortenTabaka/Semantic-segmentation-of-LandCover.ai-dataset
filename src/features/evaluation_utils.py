import json
import os
from collections import defaultdict
from json import dump
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.features.future_tensorflow import IoU


class HistoryUtilities:
    def __init__(self):
        """
        Utilities for saving history or loading it from file.
        """
        pass

    def dump_model_history_to_file(
        self,
        tensorflow_model_history: tf.keras.callbacks.History,
        folder_path: str,
        file_name: str,
    ):
        """
        Saves model history generated during training.
        Args:
            tensorflow_model_history: Tensorflow history; history = model.fit(...)
            folder_path: path to data folder
            file_name: history filename
        """
        self.create_folders([folder_path])

        if folder_path[-1] != "/":
            folder_path += "/"

        history = tensorflow_model_history.history
        dump(history, open(folder_path + file_name, "w"))

    @staticmethod
    def load_model_history_from_filepath(filepath) -> dict:
        return json.load(open(filepath, "r"))

    @staticmethod
    def load_model_history_from_folder_path(folder_path: str, file_name: str) -> dict:
        if folder_path[-1] != "/":
            folder_path += "/"
        return json.load(open(folder_path + file_name, "r"))

    @staticmethod
    def create_folders(paths: List[str]):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)


class History:
    def __init__(
        self,
        tensorflow_model_history: List[tf.keras.callbacks.History],
    ):
        """
        Class representing single or multiple tensorflow training record.

        Args:
            tensorflow_model_history: record of the training;
                training_history = model.fit(...)
        """
        self.multiple_history_files = tensorflow_model_history

    def display_history_plots(
        self,
        save_folder_path: str = None,
    ):
        """
        Displays plots for train and validation values.
        Args:
            save_folder_path: str: path to folder where images should be saved
        """
        all_history = self.merge_multiple_histories()

        if save_folder_path:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)

            if save_folder_path[-1] != "/":
                save_folder_path += "/"

        try:
            training_keys = []
            # validation_keys = []

            for key in all_history.keys():
                if "val" not in key:
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
                if save_folder_path:
                    plt.savefig(save_folder_path + f"{key}.jpg")
                plt.show()

        except AttributeError as err:
            print(
                "Validate if passed arguments are correct."
                "\nPython dictionary of model history or its filepath to textfile must be passed."
                f"\n{err}"
            )
            raise err

    def merge_multiple_histories(self):
        """
        Merges multiple multiple_history_files into one.

        Returns: Merged multiple_history_files according to key.
        """
        added_histories = []
        for history in self.multiple_history_files:
            added_histories.append(history.history)

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

    def display_confusion_matrix(self, class_names: List[str]):
        df_matrix = np.array(self.get_dataframe())

        # Normalize the confusion matrix
        confusion_matrix = (
            df_matrix.astype("float") / df_matrix.sum(axis=1)[:, np.newaxis]
        )

        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(confusion_matrix, cmap="Blues")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Add labels to the x-axis and y-axis
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=14)
        ax.set_yticklabels(class_names, fontsize=14)

        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = confusion_matrix.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j,
                    i,
                    "{:.2f}".format(confusion_matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black",
                    fontsize=14,
                )

        # Add title and axis labels
        ax.set_title("Confusion Matrix", fontsize=14)
        ax.set_xlabel("Predicted label", fontsize=14)
        ax.set_ylabel("True label", fontsize=14)

        # Show the figure
        plt.show()

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

        df = pd.DataFrame(
            iou, index=[i for i in range(self.number_of_classes)], columns=["IoU score"]
        )
        if save_directory:
            if save_directory[-1] != "/":
                save_directory += "/"
            save_path = save_directory + "iou_for_every_class.csv"
            df.to_csv(save_path)
            print(f"CSV saved to {save_path}")
        else:
            print("CSV won't be saved")

        return df
