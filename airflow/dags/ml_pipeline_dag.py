from datetime import datetime

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.edgemodifier import Label
import os

@dag(
    dag_id="ml_pipeline",
    schedule="@hourly",
    start_date=datetime(2023, 4, 14),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"],
    max_active_runs=1,
)

def ml_pipeline():
    @task.virtualenv(
        task_id="run_feature_pipeline",
        requirements=[
            "--trusted-host my-private-pypi",
            "--index-url http://my-private-pypi:8080/simple/",
            "feature_pipeline",  # 指定版本
        ],
        python_version="3.10",
        multiple_outputs=True,
        system_site_packages=True,
    )
    def run_feature_pipeline(
        config: dict
    ) -> dict:
        """
        Run the feature pipeline.

        Args:
            export_end_reference_datetime: The end reference datetime of the export window. If None, the current time is used.
                Because the data is always delayed with "days_delay" days, this date is used only as a reference point.
                The real extracted window will be computed as [export_end_reference_datetime - days_delay - days_export, export_end_reference_datetime - days_delay].

            days_delay : int
                Data has a delay of N days. Thus, we have to shift our window with N days.

            days_export : int
                The number of days to export.

            url : str
                URL to the raw data.

            feature_group_version : int
                Version of the feature store feature group to use.

        Returns:
            Metadata of the feature pipeline run.
        """

        from datetime import datetime

        from feature_pipeline import utils, pipeline

        logger = utils.get_logger(__name__)
        logger.info(f"Running feature pipeline with config: {config}")
        return pipeline.run(
            config=config
        )

    @task.virtualenv(
        task_id="create_feature_view",
        requirements=[
            "--trusted-host my-private-pypi",
            "--index-url http://my-private-pypi:8080/simple/",
            "feature_pipeline",
        ],
        python_version="3.10",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def create_feature_view(config: dict) -> dict:
        """
        This function creates a feature view based on the feature pipeline computations. The feature view
        is created using the feature group version from the feature pipeline metadata.
        """

        from feature_pipeline import utils, feature_view
        return feature_view.create(
            config=config
        )

    @task.virtualenv(
        task_id="run_hyperparameter_tuning",
        requirements=[
            "--trusted-host my-private-pypi",
            "--index-url http://my-private-pypi:8080/simple/",
            "training_pipeline",
        ],
        python_version="3.10",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def run_hyperparameter_tuning(config: dict,feature_view_metadata: dict) -> dict:
        """
        This function runs hyperparameter tuning for the training pipeline.
        The feature store feature view version and training dataset version are passed
        based on the results from the create_feature_view task.
        """

        from training_pipeline import hyperparameter_tuning

        return hyperparameter_tuning.run(
            config=config,
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
        )

    @task.virtualenv(
        task_id="upload_best_config",
        requirements=[
            "--trusted-host my-private-pypi",
            "--index-url http://my-private-pypi:8080/simple/",
            "training_pipeline",
        ],
        python_version="3.10",
        multiple_outputs=False,
        system_site_packages=False,
    )
    def upload_best_config(last_sweep_metadata: dict):
        """
        Upload the best config to W&B ML platform found in the hyperparameter tuning step
        based on the given sweep id.
        """

        from training_pipeline import best_config

        best_config.upload(sweep_id=last_sweep_metadata["sweep_id"])

    @task.virtualenv(
        task_id="train_from_best_config",
        requirements=[
            "--trusted-host my-private-pypi",
            "--index-url http://my-private-pypi:8080/simple/",
            "training_pipeline",
        ],
        python_version="3.10",
        multiple_outputs=True,
        system_site_packages=False,
        trigger_rule=TriggerRule.ALL_DONE,
    )
    def train_from_best_config(feature_view_metadata: dict) -> dict:
        """Trains model from the best config found in hyperparameter tuning.

        Args:
            feature_view_metadata (dict): Contains feature store feature view and training dataset version.

        Returns:
            metadata from the training run
        """

        from training_pipeline import train
        from training_pipeline.modules import utils

        has_best_config = utils.check_if_artifact_exists("best_config")
        if has_best_config is False:
            raise RuntimeError(
                "No best config found. Please run hyperparameter tuning first."
            )

        return train.from_best_config(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
        )

    # @task.virtualenv(
    #     task_id="compute_monitoring",
    #     requirements=[
    #         "--trusted-host my-private-pypi",
    #         "--index-url http://my-private-pypi:8080/simple/",
    #         "batch_prediction_pipeline",
    #     ],
    #     python_version="3.10",
    #     multiple_outputs=True,
    #     system_site_packages=False,
    #     trigger_rule=TriggerRule.ALL_DONE,
    # )
    # def compute_monitoring(feature_view_metadata: dict):
    #     """Compute monitoring metrics for newly obbserved data.

    #     Args:
    #         feature_view_metadata: metadata containing the version of the feature store feature view version.
    #     """

    #     from batch_prediction_pipeline import monitoring

    #     monitoring.compute(
    #         feature_view_version=feature_view_metadata["feature_view_version"],
    #     )

    @task.virtualenv(
        task_id="batch_predict",
        requirements=[
            "--trusted-host my-private-pypi",
            "--index-url http://my-private-pypi:8080/simple/",
            "batch_prediction_pipeline",
        ],
        python_version="3.10",
        multiple_outputs=True,
        system_site_packages=False,
        trigger_rule=TriggerRule.ALL_DONE,
    )
    def batch_predict(
        feature_view_metadata: dict,
        train_metadata: dict,
        feature_pipeline_metadata: dict,
        fh: int = 24,
    ):
        """
        This is the function that runs the batch prediction pipeline

        Args:
            feature_view_metadata (dict):  the metadata from the create feature view task
            train_metadata (dict): the metadata from the training pipeline task
            feature_pipeline_metadata (dict): the metadata from the feature pipeline task
            fh (int, optional): forecast horizon. Defaults to 24.
        """

        from datetime import datetime
        from batch_prediction_pipeline import batch

        batch.predict(
            fh=1,
            feature_view_version=feature_view_metadata["feature_view_version"],
            model_version=train_metadata["model_version"],
        )

    @task.branch(task_id="if_run_hyperparameter_tuning_branching")
    def if_run_hyperparameter_tuning_branching(run_hyperparameter_tuning: bool) -> bool:
        """Task used to branch between hyperparameter tuning and skipping it."""
        if run_hyperparameter_tuning is True:
            return ["branch_run_hyperparameter_tuning"]
        else:
            return ["branch_skip_hyperparameter_tuning"]

    # Define empty operators used for branching between hyperparameter tuning and skipping it.
    branch_run_hyperparameter_tuning_operator = EmptyOperator(
        task_id="branch_run_hyperparameter_tuning"
    )
    branch_skip_hyperparameter_tuning_operator = EmptyOperator(
        task_id="branch_skip_hyperparameter_tuning"
    )

    # Define Airflow variables.
    feature_pipeline_config = Variable.get("feature_pipeline", deserialize_json=True)
    should_run_hyperparameter_tuning = (
        Variable.get(
            "ml_pipeline_should_run_hyperparameter_tuning", default_var="True"
        )
        == "True"
    )
    training_pipeline_config = Variable.get("training_pipeline", deserialize_json=True)
    # Feature pipeline
    feature_pipeline_metadata = run_feature_pipeline(
        # export_end_reference_datetime="{{ dag_run.logical_date }}",
        config=feature_pipeline_config
    )
    feature_view_metadata = create_feature_view(config=feature_pipeline_config)

    # Training pipeline
    if_run_hyperparameter_tuning_branch = if_run_hyperparameter_tuning_branching(
        should_run_hyperparameter_tuning
    )
    last_sweep_metadata = run_hyperparameter_tuning(training_pipeline_config, feature_view_metadata)
    upload_best_model_step = upload_best_config(last_sweep_metadata)
    train_metadata = train_from_best_config(feature_view_metadata)

    # Batch prediction pipeline
    # compute_monitoring_step = compute_monitoring(feature_view_metadata)
    batch_predict_step = batch_predict(
        feature_view_metadata, train_metadata, feature_pipeline_metadata
    )

    # Define DAG structure.
    (
        feature_view_metadata
        >> if_run_hyperparameter_tuning_branch
        >> [
            if_run_hyperparameter_tuning_branch
            >> Label("Run HPO")
            >> branch_run_hyperparameter_tuning_operator
            >> last_sweep_metadata
            >> upload_best_model_step,
            if_run_hyperparameter_tuning_branch
            >> Label("Skip HPO")
            >> branch_skip_hyperparameter_tuning_operator,
        ]
        >> train_metadata
        # >> compute_monitoring_step
        >> batch_predict_step
    )


ml_pipeline()
