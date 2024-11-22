from typing import List
import requests

import pandas as pd
import plotly.graph_objects as go
import logging
from settings import API_URL
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 添加 StreamHandler 來輸出到 stdout
    ]
)


def build_data_plot(area: int):
    """
    Build plotly graph for data.
    """
    logging.info(f"Building data plot for area {area}")
    logging.info(f"API URL: {API_URL}")
    # Get predictions from API.
    response = requests.get(
        f"{API_URL}/predictions/{area}", verify=False
    )
    if response.status_code != 200:
        # If the response is invalid, build empty dataframes in the proper format.
        train_df = build_dataframe([], [])
        preds_df = build_dataframe([], [])

        title = "NO DATA AVAILABLE FOR THE GIVEN AREA AND CONSUMER TYPE"
    else:
        json_response = response.json()

        # 添加這些日誌
        logging.info(f"API Response: {json_response}")

        # Build DataFrames for plotting.
        datetime_utc = json_response.get("datetime_utc")
        aqi = json_response.get("aqi")
        pred_datetime_utc = json_response.get("preds_datetime_utc")
        pred_aqi = json_response.get("preds_aqi")

        # 檢查數據是否為空
        logging.info(f"Data received - datetime_utc: {datetime_utc}, aqi: {aqi}")
        logging.info(f"Predictions - pred_datetime_utc: {pred_datetime_utc}, pred_aqi: {pred_aqi}")

        train_df = build_dataframe(datetime_utc, aqi)
        preds_df = build_dataframe(pred_datetime_utc, pred_aqi)

        title = "AQI prediction for specific area"

    # Create plot.
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=16),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Datetime UTC")
    fig.update_yaxes(title_text="AQI prediction")
    fig.add_scatter(
        x=train_df["datetime_utc"],
        y=train_df["aqi"],
        name="Observations",
        line=dict(color="#C4B6B6"),
        hovertemplate="<br>".join(["Datetime: %{x}", "AQI: %{y}"]),
    )
    fig.add_scatter(
        x=preds_df["datetime_utc"],
        y=preds_df["aqi"],
        name="Predictions",
        line=dict(color="#FFC703"),
        hovertemplate="<br>".join(["Datetime: %{x}", "AQI: %{y}"]),
    )

    return fig


def build_dataframe(datetime_utc: List[int], aqi_values: List[float]):
    """
    Build DataFrame for plotting from timestamps and energy consumption values.

    Args:
        datetime_utc (List[int]): list of timestamp values in UTC (in seconds)
        values (List[float]): list of energy consumption values
    """
    df = pd.DataFrame(
        list(zip(datetime_utc, aqi_values)),
        columns=["datetime_utc", "aqi"],
    )
    # 將秒級的 timestamp 轉換為 datetime
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], unit='s')

    # Resample to hourly frequency to make the data continuous.
    df = df.set_index("datetime_utc")
    df = df.resample("H").asfreq()
    df = df.reset_index()

    return df
