import os
from typing import List

from yaml import dump, safe_load


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


def save_model_version_to_yaml(
    model_architecture: str,
    version: str,
    revision: str,
    batch_size: int,
    input_image_height: str,
    input_image_width: str,
    number_of_classes: int,
    pretrained_weights: str,
    were_layers_weights_frozen: bool,
    last_layer_frozen: int,
    activation_used: str,
    project_version_of_deeplab: str,
    output_stride: int,
    optimizer: str,
    loss_function: str,
    metrics: List[str],
):
    config_dict = {
        "model_architecture": model_architecture,
        "version": version,
        "revision": revision,
        "dataset_parameters": {
            "input_image_height": input_image_height,
            "input_image_width": input_image_width,
            "number_of_classes": number_of_classes,
            "batch_size": batch_size
        },
        "model_build_parameters": {
            "input_image_height": input_image_height,
            "input_image_width": input_image_width,
            "number_of_classes": number_of_classes,
            "pretrained_weights": pretrained_weights,
            "were_layers_weights_frozen": were_layers_weights_frozen,
            "last_layer_frozen": last_layer_frozen,
            "activation_used": activation_used,
            "project_version_of_deeplab": project_version_of_deeplab,
            "output_stride": output_stride,
        },
        "model_compile_parameters": {
            "optimizer": optimizer,
            "loss_function": loss_function,
            "metrics": metrics,
        }
    }

    yaml_filepath = get_absolute_path_to_project_location("models/model_weights.yaml")
    with open(yaml_filepath, "r") as f:
        existing_model_revisions = safe_load(f)

    existing_model_revisions.update(config_dict)

    with open(yaml_filepath, "w") as f:
        dump(existing_model_revisions, f)
