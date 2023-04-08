from pathlib import Path

import typer

from src.features.utils import get_absolute_path_to_project_location
from src.pipelines.prediction_processor import PredictionPipeline


def main(
    model_revision: str = typer.Option(
        default="deeplabv3plus_v5.10.2",
        help="Choose model revision for predictions",
    ),
    input_folder: Path = typer.Option(
        get_absolute_path_to_project_location("models/custom_data/input"),
        help='Default: "models/custom_data/input". '
        "Folder with images for which predictions should be made.",
        file_okay=False,
        dir_okay=True,
        show_default=False,
    ),
    output_folder: Path = typer.Option(
        default=get_absolute_path_to_project_location("models/custom_data/output"),
        help='Default: "models/custom_data/output". Folder with predicted masks.',
        file_okay=False,
        dir_okay=True,
        show_default=False,
    ),
) -> None:
    """
    Run the prediction pipeline.

    Parameters:
    -----------
    model_revision : str, optional
        Choose model revision for predictions. Default is "deeplabv3plus_v5.10.2".
    input_folder : Path, optional
        Folder with images for which predictions should be made. Default is "models/custom_data/input".
    output_folder : Path, optional
        Folder with predicted masks. Default is "models/custom_data/output".

    Returns:
    --------
    None
    """

    PredictionPipeline(
        model_revision=model_revision,
        input_folder=input_folder,
        output_folder=output_folder,
    ).process()


if __name__ == "__main__":
    typer.run(main)
