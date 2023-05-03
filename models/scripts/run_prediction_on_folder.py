from enum import Enum
from pathlib import Path

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
    sp_post_processing: bool = typer.Option(False),
    sp_count: int = typer.Option(200, min=0),
    sp_compactness: float = typer.Option(10, min=0),
    sp_thresh: float = typer.Option(0.7, min=0, max=1),
    sp_class_balance: bool = typer.Option(False),
    border_sp: bool = typer.Option(
        True, help="If should post-process tile boundaries with SuperPixels algorithm"
    ),
    border_sp_count: int = typer.Option(
        50, min=0, help="Will be multiplied by number of borders in single strip"
    ),
    border_compactness: float = typer.Option(10, min=0),
    border_sp_thresh: float = typer.Option(0.3, min=0, max=1),
    border_sp_class_balance: bool = typer.Option(True),
    border_sp_pixel_range: int = typer.Option(
        50,
        min=1,
        help="Strip width in single direction. Stripe width will be double of the value.",
    ),
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
        tiles_superpixel_postprocessing=sp_post_processing,
        number_of_superpixels=sp_count,
        compactness=sp_compactness,
        superpixel_threshold=sp_thresh,
        sp_class_balance=sp_class_balance,
        border_sp=border_sp,
        border_sp_count=border_sp_count,
        border_compactness=border_compactness,
        border_sp_thresh=border_sp_thresh,
        border_sp_class_balance=border_sp_class_balance,
        border_sp_pixel_range=border_sp_pixel_range,
    ).process(clear_cache)


if __name__ == "__main__":
    typer.run(main)
