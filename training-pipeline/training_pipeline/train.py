import json
from collections import OrderedDict
import os
from pathlib import Path
from typing import OrderedDict as OrderedDictType, Optional, Tuple

import fire
import hopsworks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
# from sktime.performance_metrics.forecasting import (
#     mean_squared_percentage_error,
#     mean_absolute_percentage_error,
# )
# from sktime.utils.plotting import plot_series


from training_pipeline.modules import utils
from training_pipeline.modules.settings import SETTINGS, OUTPUT_DIR
from training_pipeline.modules.feature_store_data_loader import load_dataset_from_feature_store
from training_pipeline.modules.models import build_model

from sklearn.metrics import mean_absolute_percentage_error

logger = utils.get_logger(__name__)


def from_best_config(
    feature_view_version: Optional[int] = None,
    training_dataset_version: Optional[int] = None,
) -> dict:
    """
    執行內容
    1. 從 feature store 中載入訓練資料
    2. 從 wandb 中載入最佳模型
    3. 訓練最佳模型
    4. 評估最佳模型
    5. 儲存最佳模型

    Args:
        feature_view_version (Optional[int], optional): feature store - feature view version.
             If none, it will try to load the version from the cached feature_view_metadata.json file. Defaults to None.
        training_dataset_version (Optional[int], optional): feature store - feature view - training dataset version.
            If none, it will try to load the version from the cached feature_view_metadata.json file. Defaults to None.

    Returns:
        dict: Dictionary containing metadata about the training experiment.
    """

    feature_view_metadata = utils.load_json("feature_view_metadata.json")
    if feature_view_version is None:
        feature_view_version = feature_view_metadata["feature_view_version"]
    if training_dataset_version is None:
        training_dataset_version = feature_view_metadata["training_dataset_version"]
    y_train, y_test, X_train, X_test = load_dataset_from_feature_store(
        feature_view_metadata=feature_view_metadata,
    )

    # 初始化 wandb 實驗
    with utils.init_wandb_run(
        name="best_model",
        job_type="train_best_model",
        group="train",
        reinit=True,
        add_timestamp_to_name=True,
    ) as run:
        logger.info("Loading training data from wandb...")
        run.use_artifact("split_train:latest")
        run.use_artifact("split_test:latest")

        logger.info("Loading best model from wandb...")
        best_config_artifact = run.use_artifact(
            "best_config:latest",
            type="model",
        )
        download_dir = best_config_artifact.download()
        config_path = Path(download_dir) / "best_config.json"
        with open(config_path) as f:
            config = json.load(f)
        run.config.update(config)

        logger.info("Building model by best config...")

        best_model = build_model(config)
        best_forecaster = train_model(best_model, y_train, X_train)

        y_pred, metrics = evaluate(best_forecaster, y_test, X_test)
        wandb.log({"test": {"model": metrics}})
        logger.info(f"Evaluation metrics: {metrics}")
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        logger.info(f"\n{comparison_df.head()}")

        # Save best model.
        save_model_path = OUTPUT_DIR / "best_model.pkl"
        utils.save_model(best_forecaster, save_model_path)
        metadata = {
            "experiment": {
                "feature_view_version": feature_view_version,
                "training_dataset_version": training_dataset_version,
            },
            "results": {"test": metrics},
        }
        artifact = wandb.Artifact(name="best_model", type="model", metadata=metadata)
        artifact.add_file(str(save_model_path))
        run.log_artifact(artifact)

        run.finish()
        artifact.wait()

    model_version = add_best_model_to_model_registry(artifact)

    metadata = {"model_version": model_version}
    utils.save_json(metadata, file_name="train_metadata.json")

    return metadata


def train_model(model, y_train: pd.DataFrame, X_train: pd.DataFrame):
    """Train the forecaster on the given training set and forecast horizon."""

    model.fit(X_train, y_train)

    return model


def evaluate(
    forecaster, y_test: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, dict]:
    """Evaluate the forecaster on the test set by computing the following metrics:
        - RMSPE
        - MAPE
        - Slices: RMSPE, MAPE

    Args:
        forecaster: model following the sklearn API
        y_test (pd.DataFrame): time series to forecast
        X_test (pd.DataFrame): exogenous variables

    Returns:
        The predictions as a pd.DataFrame and a dict of metrics.
    """

    y_pred = forecaster.predict(X=X_test)

    # Compute aggregated metrics.
    results = dict()
    mape = mean_absolute_percentage_error(y_test, y_pred)
    results["MAPE"] = mape

    return y_pred, results




def add_best_model_to_model_registry(best_model_artifact: wandb.Artifact) -> int:
    """Adds the best model artifact to the model registry."""

    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )

    # Upload the model to the Hopsworks model registry.
    best_model_dir = best_model_artifact.download()
    best_model_path = Path(best_model_dir) / "best_model.pkl"
    best_model_metrics = best_model_artifact.metadata["results"]["test"]

    mr = project.get_model_registry()
    py_model = mr.python.create_model("best_model", metrics=best_model_metrics)
    py_model.save(str(best_model_path))  # 將 Path 對象轉換為字符串

    return py_model.version


if __name__ == "__main__":
    fire.Fire(from_best_config)