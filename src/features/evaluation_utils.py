from __future__ import annotations

import json

from collections import defaultdict
from json import dump
import matplotlib.pyplot as plt


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

    @staticmethod
    def display_single_history_plots(model_history: None | dict = None, history_path: None | str = None):

        if history_path:
            model_history = HistoryUtilities(history_path).load_model_history_from_file()

        try:
            for key in model_history.keys():
                plt.plot(model_history[key])
                plt.title(f"Training {key} (Epoch)")
                plt.ylabel(f"{key}")
                plt.xlabel("Epoch")
                plt.show()

        except AttributeError as err:
            print("Validate if passed arguments are correct."
                  "\nPython dictionary of model history or its filepath to JSON must be passed."
                  f"\n{err}")

    @staticmethod
    def merge_multiple_histories(
            histories: None | list[dict] = None,
            histories_filepaths: None | list[str] = None
    ) -> defaultdict[list]:

        if histories_filepaths and not histories:
            histories = []
            for filepath in histories_filepaths:
                history = HistoryUtilities(filepath).load_model_history_from_file()
                histories.append(history)

        merged = defaultdict(list)

        if len(histories) > 1:
            for single_history in histories:
                for key, value in single_history.items():
                    merged[key].append(value)

        return merged

