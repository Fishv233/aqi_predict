from typing import Tuple
import hopsworks
import pandas as pd
import wandb
import numpy as np
from training_pipeline.modules.utils import init_wandb_run
from training_pipeline.modules.data_transform import XGBDataPreprocessor
from training_pipeline.modules.settings import SETTINGS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from training_pipeline.modules.utils import get_logger

logger = get_logger(__name__)

def load_dataset_from_feature_store(
    feature_view_metadata: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Args:
    Returns:
        Train and test splits loaded from the feature store as pandas dataframes.
    """
    # login to hopsworks
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )
    fs = project.get_feature_store()
    # get data from hopsworks feature store and record metadata of feature view
    with init_wandb_run(
        name="load_training_data", job_type="load_feature_view", group="dataset"
    ) as run:
        feature_view = fs.get_feature_view(
            name=feature_view_metadata["project_name"],
            version=feature_view_metadata["feature_view_version"],
        )
        get_feature_group = fs.get_feature_group(
            name="aqi_feature_group", 
            version=1
        )
        
        data = get_feature_group.read()
        # organize metadata of feature view

        raw_data_at = wandb.Artifact(
            name=f"{feature_view_metadata['project_name']}feature_view",
            type="feature_view",
        )
        run.log_artifact(raw_data_at)

        run.finish()

    # data_transform
    with init_wandb_run(
        name="train_test_split", job_type="prepare_dataset", group="dataset"
    ) as run:
        run.use_artifact(f"{feature_view_metadata['project_name']}feature_view:latest")

        y_train, y_test, X_train, X_test = data_transform(data)

        for split in ["train", "test"]:
            split_X = locals()[f"X_{split}"]
            split_y = locals()[f"y_{split}"]

            split_metadata = {
                "dataset_size": split_X.shape[0],
                "X_shape": split_X.shape,
                "y_shape": split_y.shape,
                "X_features": split_X.shape[-1],
                "y_features": split_y.shape[-1] if split_y.ndim > 1 else 1,
            }
            artifact = wandb.Artifact(
                name=f"split_{split}",
                type="split",
                metadata=split_metadata,
            )
            run.log_artifact(artifact)

        run.finish()

    return y_train, y_test, X_train, X_test


def data_transform(
    data: pd.DataFrame,
    sequence_length: int = 20,
    prediction_steps: int = 1,
    scale_data: bool = False,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessor = XGBDataPreprocessor(sequence_length, prediction_steps, scale_data)
    X, y = preprocessor.prepare_data(data)
    X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y, test_size=test_size)

    return y_train, y_test, X_train, X_test