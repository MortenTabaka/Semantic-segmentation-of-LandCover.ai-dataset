from pathlib import Path
from enum import Enum

import typer

from src.features.utils import get_absolute_path_to_project_location
from src.pipelines.prediction_processor import PredictionPipeline


class TypeOfWeightsToLoad(str, Enum):
    miou = "miou"
    loss = "loss"


class AvailableRevisions(str, Enum):
    version_5_10_2 = "deeplabv3plus_v5.10.2"
    version_10_0_1 = "deeplabv3plus_v10.0.1"
    version_12_1_2 = "deeplabv3plus_v12.1.2"


def main(
    model_revision: AvailableRevisions = typer.Option(
        default=AvailableRevisions.version_5_10_2,
        help="Choose model revision for predictions.",
        show_choices=True,
    ),
    weights: TypeOfWeightsToLoad = typer.Option(
        default=TypeOfWeightsToLoad.miou,
        help="Pick which weights load",
        show_choices=True,
    ),
    tiles_superpixel_postprocessing: bool = typer.Option(True),
    number_of_superpixels: int = typer.Option(200, min=0),
    compactness: float = typer.Option(10, min=0),
    superpixel_threshold: float = typer.Option(0.7, min=0, max=1),
    postprocess_boundaries: bool = typer.Option(True),
    clear_cache: bool = typer.Option(True),
    input_folder: Path = typer.Option(
        get_absolute_path_to_project_location("models/custom_data/input"),
        help='Default: "models/custom_data/input". '
        "Folder with input images (JPG/PNG/TIFF)",
        file_okay=False,
        dir_okay=True,
        show_default=False,
    ),
    output_folder: Path = typer.Option(
        default=get_absolute_path_to_project_location("models/custom_data/output"),
        help='Default: "models/custom_data/output". Output predictions will be saved here.',
        file_okay=False,
        dir_okay=True,
        show_default=False,
    ),
) -> None:
    PredictionPipeline(
        model_revision=model_revision,
        input_folder=input_folder,
        output_folder=output_folder,
        which_metric_best_weights_to_load=weights,
        tiles_superpixel_postprocessing=tiles_superpixel_postprocessing,
        number_of_superpixels=number_of_superpixels,
        compactness=compactness,
        superpixel_threshold=superpixel_threshold,
    ).process(postprocess_boundaries, clear_cache)


if __name__ == "__main__":
    typer.run(main)
