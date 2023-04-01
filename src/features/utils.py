import os
from typing import List, Dict, Tuple, Union
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
    model_build_params: Dict[str, Union[int, Tuple[int, int, int], str, None, float]],
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
        input_image_width:
        input_image_height:
        model_build_params: params needed to build model with
            e.g. src.models.architectures.deeplabv3plus
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
        "model_build_parameters": model_build_params,
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

    model_key = f"{model_name}_v{revision}"

    yaml_filepath = get_absolute_path_to_project_location(
        "models/models_revisions.yaml"
    )

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

    with open(yaml_filepath, "w") as f:
        dump(existing_models_revisions, f, default_flow_style=False, sort_keys=False)

    return model_key


def get_model_build_params_for_revision(model_key):
    data = load_data_for_revision(model_key)

    try:
        pretrained_weights = data["model_build_parameters"]["pretrained_weights"]
        second_input = data["model_build_parameters"]["second_input"]
        input_shape = (
            data["model_build_parameters"]["input_shape"]["input_image_height"],
            data["model_build_parameters"]["input_shape"]["input_image_width"],
            data["model_build_parameters"]["input_shape"]["channels"],
        )
        num_classes = data["model_build_parameters"]["num_classes"]
        backbone = data["model_build_parameters"]["backbone"]
        output_stride = data["model_build_parameters"]["output_stride"]
        alpha = data["model_build_parameters"]["alpha"]
        activation = data["model_build_parameters"]["activation"]

        model_build_parameters = [pretrained_weights, second_input, input_shape, num_classes, backbone,
                                  output_stride, alpha, activation, ]
    except KeyError as e:
        raise ValueError(
            f"YAML file does not contain expected data: {e}"
        )

    return model_build_parameters


def get_revision_model_architecture(model_key: str):
    data = load_data_for_revision(model_key)
    return data["model_name"]


def load_data_for_revision(model_key):
    yaml_filepath = get_absolute_path_to_project_location(
        "models/models_revisions.yaml"
    )
    if not os.path.exists(yaml_filepath):
        raise ValueError(f"YAML file {yaml_filepath} does not exist")

    with open(yaml_filepath, "r") as f:
        existing_models_revisions = safe_load(f)

    data = existing_models_revisions.get(model_key, {})

    if not data:
        raise ValueError(
            f"YAML file {yaml_filepath} does not contain expected data."
        )
    return data
