from joblib import load
from loguru import logger
import matplotlib.pyplot as plt
import typer
from typing_extensions import Annotated

from mnist_classification.config import ARTIFACT_DIR, FIGURES_DIR
from modeling.train import handle_saved_file_name


def save_fig(
    plot_name: str,
    tight_layout: bool = False,
    fig_extention: str = "png",
    resolution: int = 300,
) -> None:
    figure_path = FIGURES_DIR / f"{plot_name}.{fig_extention}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(figure_path, format=fig_extention, dpi=resolution)
    plt.close()


def load_pr_curve_data(model_name, target_digit):
    filename = handle_saved_file_name("pr_curve", model_name, target_digit) + ".joblib"
    path = ARTIFACT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"PR curve filename not found at {path}")
    data = load(path)
    return data["precisions"], data["recalls"], data["thresholds"]


def plot_pr_curve_vs_threshold(model_name: str, target_digit: int, threshold: int):
    precisions, recalls, thresholds = load_pr_curve_data(model_name, target_digit)
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.vlines(threshold, 0, 1.0, "m", "dotted", label="threshold")
    idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
    plt.plot(thresholds[idx], precisions[idx], "bo")
    plt.plot(thresholds[idx], recalls[idx], "go")
    plt.axis((-50000, 50000, 0, 1))
    plt.grid()
    plt.xlabel("Threshold")
    plt.legend(loc="center right")
    save_fig("precision_recall_vs_threshold_plot")
    logger.info("Successfully saved the figure")


app = typer.Typer()


@app.command()
def main(
    model_name: Annotated[
        str,
        typer.Option(
            "--model-name",
            "-m",
            case_sensitive=False,
            help="Choose the classifier model",
        ),
    ],
    target_digit: Annotated[
        int, typer.Option("--digit", "-d", help="Digit to classify (0-9)")
    ] = 5,
    threshold: Annotated[
        int, typer.Option("--threshold", "-t", help="Threshold number")
    ] = 3000,
):
    logger.info(f"Plotting PR curve for {model_name}, digit {target_digit}")
    plot_pr_curve_vs_threshold(model_name, target_digit, threshold)


if __name__ == "__main__":
    app()
