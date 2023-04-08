from pathlib import Path

import typer

from src.features.utils import get_absolute_path_to_project_location
from src.pipelines.prediction_processor import PredictionPipeline


def main(
    model_revision: str = typer.Option(
        default="deeplabv3plus_v5.10.2",
        help="Choose model revision for predictions",
    ),
    checkpoint_to_load: Path = typer.Option(
        default=None,
        help='Use to run predictor with chosen checkpoint weights. Default: None.',
        file_okay=False,
        dir_okay=True,
        show_default=False,
    ),
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
    clear_cache: bool = typer.Option(True),
) -> None:

    PredictionPipeline(
        model_revision=model_revision,
        input_folder=input_folder,
        output_folder=output_folder,
        path_to_local_checkpoint=checkpoint_to_load
    ).process(clear_cache)


if __name__ == "__main__":
    typer.run(main)
