import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from typing import Tuple
from training_pipeline.modules.utils import get_logger

logger = get_logger(__name__)
class BaseDataPreprocessor:
    def __init__(self, sequence_length=20, prediction_steps=1, scale_data=False):
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.scale_data = scale_data
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if scale_data else None

    def inverse_transform(self, data):
        if self.scale_data:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

    def split_train_test(self, X, y, test_size=0.2):
        split_index = int(len(X) * (1 - test_size))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def _transform_signal_valve(self, raw_data, columns=['signal_valve']):
        process_data = raw_data[columns].values.flatten() 
        # 初始化結果數組
        if not isinstance(process_data, pd.Series):
            process_data = pd.Series(process_data)
        # 計算差分，用於檢測連續性
        diff = process_data.diff()
        # 初始化結果序列
        result = pd.Series(np.zeros(len(process_data), dtype=int))
        # 初始化計數器
        counter = 1

        # 遍歷數據
        for i in range(len(process_data)):
            if i == 0 or diff.iloc[i] != 0:
                # 如果是第一個元素或者不連續，重置計數器
                counter = 1

            # 賦值並增加計數器
            result.iloc[i] = counter
            counter += 1

        raw_data['signal_valve'] = result
        raw_data.loc[raw_data[columns[0]] == 0, 'signal_valve'] *= -1
        training_data = raw_data.drop(columns=columns)
        return training_data, raw_data


class XGBDataPreprocessor(BaseDataPreprocessor):

    
    def prepare_data(self, raw_data, window_size=4) -> Tuple[np.ndarray, np.ndarray]:
        df = raw_data
        df.drop(columns=['pollutant'], inplace=True)

        df['ts'] = pd.to_datetime(df['ts'])
        df = df.sort_values(['sitename', 'ts'])
        # 先將需要的欄位轉換為數值型
        
        numeric_cols = ['aqi', 'so2', 'co', 'o3', 'pm10', 'no2', 'windspeed', 'pm10_avg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' 會將無法轉換的值設為 NaN

        for site in df['sitename'].unique():
            site_mask = df['sitename'] == site
            site_data = df[site_mask].copy()
            # 確保資料按時間排序
            site_data = site_data.sort_values('ts')
            # 對指定的數值欄位進行線性插值
            for col in numeric_cols:
                site_data[col] = site_data[col].interpolate(method='linear', limit_direction='both')
            # 更新原始資料框
            df.loc[site_mask] = site_data
        
        features = []
        targets = []
        
        # 排除不需要的欄位
        exclude_columns = ['ts', 'sitename']
        target_column = 'aqi'
        
        # 對每個測站分別處理
        for site in df['sitename'].unique():
            site_data = df[df['sitename'] == site]
            site_data.drop_duplicates(subset=['ts'], inplace=True)
            site_data.drop(columns=exclude_columns, inplace=True)
            # 使用可配置的滑動窗口大小
            for i in range(len(site_data) - window_size):
                # 取得連續時間點的數據
                window_data = site_data.iloc[i:i+window_size]
                
                # 收集前 window_size-1 個時間點的所有特徵
                feature = []
                for t in range(window_size-1):  # 收集除了最後一個時間點以外的所有數據
                    feature.extend(window_data.iloc[t].values.tolist())
                
                # 目標是最後一個時間點的 AQI
                target = window_data.iloc[window_size-1][target_column]
                
                features.append(feature)
                targets.append(target)
        
        return np.array(features), np.array(targets)
