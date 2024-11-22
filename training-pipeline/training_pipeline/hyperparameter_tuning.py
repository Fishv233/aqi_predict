from functools import partial
from typing import Optional, Dict, Any
import json
import fire
import wandb
from sklearn.model_selection import TimeSeriesSplit
from training_pipeline.modules.feature_store_data_loader import load_dataset_from_feature_store
from training_pipeline.modules import utils
from training_pipeline.modules.utils import init_wandb_run
from training_pipeline.modules.models import build_model
from training_pipeline.modules.settings import SETTINGS
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np


logger = utils.get_logger(__name__)


def load_config():
    with open('training-pipeline/training_pipeline/configs/config_hyperparameter.json', 'r') as f:
        return json.load(f)


class HyperparameterOptimizer:
    def __init__(self, feature_view_version: Optional[int] = None,
                 training_dataset_version: Optional[int] = None,
                 config: Dict[str, Any] = None):
        self.feature_view_version = feature_view_version
        self.training_dataset_version = training_dataset_version
        self.y_train = None
        self.X_train = None
        self.config = config

    def run_optimization(self) -> Dict[str, Any]:
        """執行超參數優化，步驟如下
        1. 載入數據->資料轉換->資料拆分
        2. 執行超參數優化
        3. 儲存優化結果
        """
        # 1.
        logger.info(f"start loading data")
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        if self.feature_view_version is None:
            self.feature_view_version = feature_view_metadata["feature_view_version"]
        if self.training_dataset_version is None:
            self.training_dataset_version = feature_view_metadata["training_dataset_version"]

        logger.info(f"start loading dataset and transform data")
        self.y_train, _, self.X_train, _ = load_dataset_from_feature_store(
            feature_view_metadata=feature_view_metadata
        )
        logger.info(f"finish loading dataset and transform data")
        logger.info(f"y_train: {self.y_train.shape}")
        logger.info(f"X_train: {self.X_train.shape}")
        # 2. 
        sweep_id = self._run_hyperparameter_optimization()
        # 3. 
        metadata = {"sweep_id": sweep_id}
        utils.save_json(metadata, file_name="last_sweep_metadata.json")

        return metadata

    def _run_hyperparameter_optimization(self) -> str:
        """使用W&B sweeps執行超參數優化"""
        hyperparameter_tuning = self.config['hyperparameter_tuning']
        # 增加sweep配置，載入超參數相關設定
        sweep_id = wandb.sweep(
            sweep=hyperparameter_tuning, project=SETTINGS["WANDB_PROJECT"]
        )

        # 從wandb中獲取agent以及對應的超參數，並使用定義的_run_sweep函數執行訓練
        wandb.agent(
            project=SETTINGS["WANDB_PROJECT"],
            sweep_id=sweep_id,
            function=partial(self._run_sweep),
        )

        return sweep_id

    def _run_sweep(self) -> None:
        """執行單次sweep，並使用wandb的artifact儲存模型及相關紀錄"""
        with init_wandb_run(
            name="experiment", job_type="hpo", group="train", add_timestamp_to_name=True
        ) as run:
            # 使用wandb的artifact載入訓練數據
            run.use_artifact("split_train:latest")

            # 使用wandb的config載入超參數
            config = dict(wandb.config)
            # 使用定義的build_model函數建立模型
            model = build_model(config)

            # 使用定義的train_model_cv函數訓練模型
            model, results = self._train_model_cv(model, self.y_train, self.X_train)
            wandb.log(results)

            metadata = {
                "experiment": run.name,
                "results": results,
                "config": config,
            }
            # 使用wandb的artifact儲存模型及相關紀錄
            artifact = wandb.Artifact(
                name="config",
                type="model",
                metadata=metadata,
            )
            run.log_artifact(artifact)

    def _train_model_cv(self, model, y, X, n_splits=3):
        """Simplified cross-validation for LightGBM model using basic KFold."""

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mape_scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.fit(X_train, y_train) 
            y_pred = model.predict(X_val)

            mape = mean_absolute_percentage_error(y_val, y_pred)
            mape_scores.append(mape)

        avg_mape = np.mean(mape_scores)
        logger.info(f"Average MAPE: {avg_mape:.2f}")

        return model, {"validation": {"MAPE": avg_mape}}


def run(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    training_dataset_version: Optional[int] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """開始執行超參數優化"""
    optimizer = HyperparameterOptimizer(feature_view_version, training_dataset_version, config)
    return optimizer.run_optimization()

if __name__ == "__main__":
    # 獲取配置
    config = load_config()
    fire.Fire(run(config=config))