import typer
from pathlib import Path

from src.features.utils import get_absolute_path_to_project_location
from src.pipelines.prediction_processor import PredictionPipeline


def main(
    model_revision: str = typer.Option(
        default="deeplabv3plus_v5.10.2",
        help="Choose model revision used for predictions",
    ),
    model_revisions: Path = typer.Option(
        get_absolute_path_to_project_location("models/models_revisions.yaml"),
        help="Path to YAML file with model revision. "
        "Use only when YAML is other than the project's default",
        file_okay=True,
        dir_okay=False,
        show_default=False,
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
):
    PredictionPipeline(
        model_revision=model_revision,
        input_folder=input_folder,
        output_folder=output_folder,
    ).process()


if __name__ == "__main__":
    typer.run(main)
