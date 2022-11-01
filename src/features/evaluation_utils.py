from __future__ import annotations

import json
from collections import defaultdict
from json import dump

import matplotlib.pyplot as plt
import numpy as np


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
        model_histories: None | list[dict] | list[str] = None,
    ):
        if all(isinstance(item, dict) for item in model_histories):
            all_history = self.merge_multiple_histories(histories=model_histories)
        elif all(isinstance(item, str) for item in model_histories):
            all_history = self.merge_multiple_histories(
                histories_filepaths=model_histories
            )
        else:
            all_history = defaultdict()
            print("History paths or dictionaries were not passed to function.")

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
                # plt.savefig(data_save_dir + f'/{name}_all.jpg')
                plt.show()

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
