import os
from typing import List, Optional, Union
from tensorflow import keras

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


def revision_a_model(
    model_architecture: str,
    revision: str,
    batch_size: int,
    input_image_height: int,
    input_image_width: int,
    number_of_classes: int,
    pretrained_weights: str,
    were_layers_weights_frozen: bool,
    last_layer_frozen: Optional[int],
    activation_used: str,
    project_version_of_deeplab: str,
    output_stride: int,
    optimizer: keras.optimizers.Optimizer = None,
    loss_function: keras.losses.Loss = None,
    metrics: Union[
        keras.metrics.Metric, str, List[Union[keras.metrics.Metric, str]]
    ] = None,
):
    config_dict = {
        "model_architecture": model_architecture,
        "revision": revision,
        "dataset_parameters": {
            "input_image_height": input_image_height,
            "input_image_width": input_image_width,
            "number_of_classes": number_of_classes,
            "batch_size": batch_size,
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
            "optimizer": str(optimizer),
            "loss_function": str(loss_function),
            "metrics": str(metrics),
        },
    }

    yaml_filepath = get_absolute_path_to_project_location(
        "models/models_revisions.yaml"
    )
    print(yaml_filepath)

    if not os.path.exists(yaml_filepath):
        # create empty file if it doesn't exist
        with open(yaml_filepath, "w") as f:
            f.write("")

    with open(yaml_filepath, "r") as f:
        existing_models_revisions = safe_load(f)

    if existing_models_revisions is None:
        existing_models_revisions = config_dict

    if (model_architecture, revision) in existing_models_revisions:
        existing_models_revisions[(model_architecture, revision)].update(config_dict)
    else:
        existing_models_revisions[(model_architecture, revision)] = config_dict

    with open(yaml_filepath, "w") as f:
        dump(existing_models_revisions, f)
