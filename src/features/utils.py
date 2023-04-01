import os
from typing import List, Optional, Union
from tensorflow import keras
from pandas import DataFrame

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
    model_name: str,
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
    optimizer: keras.optimizers.Optimizer,
    loss_function: keras.losses.Loss,
    initial_learning_rate: float,
    final_learning_rate: Union[float, None],
    metrics: Union[keras.metrics.Metric, str, List[Union[keras.metrics.Metric, str]]],
):
    """
    Utility to create yaml file for model revisions
    Args:
        model_name: name of model got with model.name
            e.g. deeplabv3plus or modified_v2_deeplabv3plus
        revision: version of model during training, e.g. 10.0.1
        batch_size: size of single batch
        pretrained_weights: weights to load to model
        were_layers_weights_frozen: should some layers be not trainable;
                must set value for border by using last_layer_frozen;
        last_layer_frozen: below that layer, all will be frozen
        activation_used: optional activation to add to the top of the network.
                One of 'softmax', 'sigmoid' or None
        project_version_of_deeplab: one of "original", "v1", "v2", "v3", "v4"
        output_stride: determines input_shape/feature_extractor_output ratio. One of {8,16}.
        input_image_width:
        input_image_height:
        optimizer:
        loss_function:
        initial_learning_rate:
        final_learning_rate: learning rate in decaying scheduler
        number_of_classes:
        metrics:
    Returns:
        model_key: f"{model_architecture}_{revision}"
    """
    config_dict = {
        "model_name": model_name,
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
            "loss_function": {
                "object": str(loss_function),
                "initial_learning_rate": initial_learning_rate,
                "final_learning_rate": final_learning_rate,
            },
            "metrics": [str(metric) for metric in metrics],
        },
    }

    model_key = f"{model_name}_{revision}"

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
        existing_models_revisions = {}

    if model_key in existing_models_revisions:
        existing_models_revisions[model_key].update(config_dict)
    else:
        existing_models_revisions[model_key] = config_dict

    sorted_models_revisions = dict(
        sorted(existing_models_revisions.items(), reverse=True)
    )

    with open(yaml_filepath, "w") as f:
        dump(sorted_models_revisions, f, default_flow_style=False)

    return model_key
