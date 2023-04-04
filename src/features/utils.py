import os

import matplotlib.pyplot as plt
import numpy as np


def get_project_root() -> str:
    """Returns the absolute path of the project root directory."""
    current_dir = os.path.abspath(os.curdir)
    while not os.path.isfile(os.path.join(current_dir, "README.md")):
        current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    return current_dir


def get_absolute_path_to_project_location(path_from_project_root: str) -> str:
    if path_from_project_root[0] == "\\":
        path_from_project_root = path_from_project_root[1:]
    return os.path.join(get_project_root(), path_from_project_root)


def generate_colormap(num_classes):
    # define the colormap
    cmap = plt.cm.get_cmap("viridis", num_classes)

    # create a list of RGB values for each class
    colormap = []
    for i in range(num_classes):
        rgb = cmap(i)[:3]
        colormap.append(rgb)

    return np.array(colormap)
