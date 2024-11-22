from typing import List, Any

from pydantic import BaseModel


class PredictionResults(BaseModel):
    datetime_utc: List[int]
    aqi: List[float]
    preds_datetime_utc: List[int]
    preds_aqi: List[float]


class MonitoringMetrics(BaseModel):
    datetime_utc: List[int]
    mape: List[float]


class MonitoringValues(BaseModel):
    y_monitoring_datetime_utc: List[int]
    y_monitoring_energy_consumption: List[float]
    predictions_monitoring_datetime_utc: List[int]
    predictions_monitoring_energy_consumptionc: List[float]
