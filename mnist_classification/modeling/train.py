from joblib import dump, load
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict, cross_val_score
import typer
from typing_extensions import Annotated

from mnist_classification.config import (
    ARTIFACT_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAN_STAT,
)


def load_mnist_data():
    """Load mnist train/test splits"""
    logger.info("Loading data from parquet files")
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "x_train.parquet").to_numpy()
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet").to_numpy().ravel()
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "x_test.parquet").to_numpy()
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet").to_numpy().ravel()
    logger.info(f"Loaded train set:{X_train.shape} and test set: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def create_binary_labels(y: np.ndarray, target_digit: int) -> np.ndarray:
    """Convert multiclass labels to binary class label (1 if target_digit otherwise 0)"""
    if not (0 <= target_digit <= 9):
        raise ValueError(f"target_digit must be between 0 and 9, got {target_digit}")
    binary_labels = y == str(target_digit)
    logger.info(f"Created binary labels for digit '{target_digit}'")
    return binary_labels


def train_sgd_binary_classifier(
    X_train: np.ndarray, y_binary: np.ndarray, random_state: int = RAN_STAT
) -> SGDClassifier:
    """Train SGD classifier for binary classification"""
    logger.info("Training SGD binary classifier....")
    sgd_clf = SGDClassifier(random_state=random_state)
    sgd_clf.fit(X_train, y_binary)
    logger.info(f"The {type(sgd_clf).__name__} training completed")
    return sgd_clf


def compute_precision_recall_curve(
    model: SGDClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 3,
    replace: bool = False,
):
    """Compute precision/recall curve using cross validation decision scores"""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pr_curve_path = ARTIFACT_DIR / "pr_curve_train.joblib"
    logger.info(f"Computing precision-recall curve with {cv_folds} folds cv....")
    if not pr_curve_path.exists() or replace:
        y_scores = cross_val_predict(
            model, X_train, y_train, cv=cv_folds, method="decision_function", n_jobs=-1
        )
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        pr_curve_data = {
            "precisions": precisions,
            "recalls": recalls,
            "thresholds": thresholds,
        }

        dump(pr_curve_data, pr_curve_path)
    else:
        pr_curve_data = load(pr_curve_path)

    return pr_curve_data


app = typer.Typer()


@app.command()
def main(
    target_digit: Annotated[
        int, typer.Option("--digit", "-d", help="Digit to classify (0-9)")
    ] = 5,
    save_model: Annotated[bool, typer.Option(help="Save trained model to disk")] = True,
    cv_folds: Annotated[
        int, typer.Option("--cv", help="Number of cross-validation folds")
    ] = 3,
):
    """Train a binary SGD classifier to detect a spesific MNIST digit"""
    logger.info("Statring model training....")
    X_train, y_train, X_test, y_test = load_mnist_data()
    y_train_binary = create_binary_labels(y_train, target_digit)
    sgd_model = train_sgd_binary_classifier(X_train, y_train_binary, 5)
    model = train_sgd_binary_classifier(X_train, y_train_binary)

    pr_curve = compute_precision_recall_curve(
        model, X_train, y_train_binary, cv_folds=cv_folds
    )
    if save_model:
        sgd_model_path = MODELS_DIR / f"sgd_digit_{target_digit}.joblib"
        pr_curve_path = ARTIFACT_DIR / f"pr_curve_{target_digit}.joblib"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        dump(sgd_model, sgd_model_path)
        dump(pr_curve, pr_curve_path)
        logger.info(f"Model saved to {sgd_model_path}")

    logger.info("Training pipline compeleted successfully")


if __name__ == "__main__":
    app()
