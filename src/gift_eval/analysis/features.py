import numpy as np
import re
from tsfeatures import tsfeatures, stl_features, entropy, hurst, lumpiness, stability

import pandas as pd

# Define periods for different frequency strings
PERIODS = {'H': 24, 'D': 7,  # hourly, daily
           'M': 12, 'Q': 4,  # monthly, quarterly
           'W': 4, 'A': 1,  # weekly, annual
           'T': 60, 'S': 60,  # minute, second
           'L': 1000, 'U': 1000, 'N': 1000}  # millisecond, microsecond, nanosecond


def infer_period(freq):
    """
    Infer the period of a time series based on its frequency string.

    Parameters:
    - freq: Frequency string (e.g., 'H', 'D', '2A-DEC').

    Returns:
    - The period as an integer.

    Raises:
    - ValueError: If the frequency is not recognized.
    """
    if '-' in freq:
        freq = freq.split('-')[0]

    if freq in PERIODS:
        return PERIODS[freq]
    elif freq.isalnum():
        pattern = r"(\d+)([a-zA-Z]+)"
        match = re.match(pattern, freq)
        repeat_count, freq_str = match.groups()
        return max(PERIODS[freq_str]//int(repeat_count), 1)
    else:
        raise ValueError(f"Frequency {freq} not recognized")


def get_ts_features(timeseries: np.ndarray, freq) -> float:
    """
    Extract time series features using the tsfeatures package.

    Parameters:
    - timeseries: A numpy array representing the time series data.
    - freq: Frequency string of the time series.

    Returns:
    - A DataFrame containing selected features: trend, seasonal_strength, entropy, hurst, lumpiness, stability.
    """
    # Create a DataFrame with a date range and the time series data
    panel = pd.DataFrame({'ds': pd.date_range(
        start='1900-01-01', periods=len(timeseries), freq=freq), 'y': timeseries})
    panel['unique_id'] = 1

    # Compute features using tsfeatures
    features_df = tsfeatures(panel, features=[
                             stl_features, entropy, hurst, lumpiness, stability], freq=infer_period(freq))

    # Ensure all required columns are present, filling missing ones with NaN
    for column in ['trend', 'seasonal_strength', 'entropy', 'hurst', 'lumpiness', 'stability']:
        if column not in features_df.columns:
            features_df[column] = np.nan
    return features_df[['trend', 'seasonal_strength', 'entropy', 'hurst', 'lumpiness', 'stability']]


if __name__ == "__main__":
    # Test the infer_period function with various frequency strings
    print(infer_period('30T'))
    print(infer_period('2A-DEC'))
    print(infer_period('H'))
    print(infer_period('A'))
    print(infer_period('2A'))
    print(infer_period('A-DEC'))
    print(infer_period('A-JAN'))
    print(infer_period('5S'))

    # Generate random time series data and test feature extraction
    timeseries = np.random.randn(100)
    print(get_ts_features(timeseries, '30T'))

    # Generate a time series with all zeros and test feature extraction
    timeseries = np.zeros(100)
    print(get_ts_features(timeseries, 'D'))

    # Generate a time series with a trend and test feature extraction
    timeseries = np.arange(100)
    print(get_ts_features(timeseries, '30T'))

    # Generate a time series with seasonality and test feature extraction
    timeseries = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    print(get_ts_features(timeseries, 'M'))

    # Generate a time series with a trend in the first half and random in the second half
    timeseries = np.concatenate([np.arange(50), np.random.randn(50)])
    print(get_ts_features(timeseries, 'H'))
