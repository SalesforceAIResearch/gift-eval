import os
import gc
import numpy as np

import ray
from ray.experimental import tqdm_ray

import pandas as pd
from pandas.tseries.frequencies import to_offset

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from functools import cached_property
from gluonts.time_feature import norm_freq_str

from gift_eval.data import Dataset
from .features import get_ts_features
from .utils import persist_analysis

from dotenv import load_dotenv
load_dotenv()

MAX_CONTEXT_LEN = 500
runtime_env = {
    'env_vars': {
        "RAY_memory_usage_threshold": "0.85"
    }
}

if not os.getenv("NUM_CPUS"):
    print("NUM_CPUS environment variable not found. Setting to 1. Set NUM_CPUS to speed up processing.")
NUM_CPUS = int(os.getenv("NUM_CPUS", "1"))


@ray.remote(num_cpus=NUM_CPUS)
def process_instance(self, test_input, test_label, dataset_freq):
    """
    Process a single time series instance to compute features.

    Parameters:
    - self: Reference to the calling object.
    - test_input: Dictionary containing the input time series data.
    - test_label: Dictionary containing the label time series data.
    - dataset_freq: Frequency of the dataset.

    Returns:
    - DataFrame containing the computed features for the time series instance.
    """
    np_inp = np.array(test_input["target"])
    np_label = np.array(test_label["target"])

    # Check if the input is 2D and trim to MAX_CONTEXT_LEN if necessary
    if len(np_inp.shape) == 2:
        if np_inp.shape[1] > MAX_CONTEXT_LEN:
            np_inp = np_inp[:, -MAX_CONTEXT_LEN:]
        np_instance = np.concatenate((np_inp, np_label), axis=1)
    else:
        if len(np_inp) > MAX_CONTEXT_LEN:
            np_inp = np_inp[-MAX_CONTEXT_LEN:]
        np_instance = np.concatenate((np_inp, np_label))

    # Compute time series features
    window_features_df = get_ts_features(
        np_instance, norm_freq_str(to_offset(dataset_freq).name))
    self.pbar.update.remote(1)
    return window_features_df


@ray.remote
def process_dataset(self, dataset, output_dir):
    """
    Process an entire dataset to compute features for each time series instance.

    Parameters:
    - self: Reference to the calling object.
    - dataset: The dataset to be processed.
    - output_dir: Directory where the processed data will be saved.

    Returns:
    - None, but updates the progress bar and persists the analysis results.
    """
    # Determine the directory for the dataset based on its term and name
    if str(dataset.term) == "Term.SHORT":
        dataset_dir = Path(os.path.join(os.path.dirname(
            output_dir), f"datasets/{dataset.name}"))
    else:
        if "/" in dataset.name:
            dataset_name, dataset_freq = dataset.name.split("/")
            dataset_name = f"{dataset_name}:{dataset.term}/{dataset_freq}"
            dataset_dir = Path(os.path.join(os.path.dirname(
                output_dir), f"datasets/{dataset_name}"))
        else:
            dataset_dir = Path(os.path.join(os.path.dirname(
                output_dir), f"datasets/{dataset.name}:{dataset.term}"))

    # Create the directory if it doesn't exist
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print("Directory created:", dataset_dir)
    else:
        print("Directory already exists:", dataset_dir)
        # Assume dataset has already been processed
        self.pbar.update.remote(dataset.hf_dataset.num_rows * dataset.windows)
        return None

    all_features_list = []
    test_data = dataset.test_data

    # Process each instance in the dataset
    features = [process_instance.remote(
        self, test_input, test_label, dataset.freq) for test_input, test_label in test_data]

    for feature in features:
        try:
            # Retrieve the result with a timeout
            result = ray.get(feature, timeout=300)  # 300 seconds timeout
            all_features_list.append(result)
        except ray.exceptions.GetTimeoutError:
            print("A task timed out and will be skipped.")
            continue  # Skip this particular instance
        except Exception as e:
            print(f"An error occurred while processing: {e}")
            continue

    gc.collect()

    # Concatenate all features and persist the analysis
    all_features_df = pd.concat(all_features_list)
    persist_analysis(all_features_df, dataset_dir)


class Analyzer():
    """
    Analyzer class to manage the analysis of multiple datasets, including feature computation and frequency distribution analysis.
    """

    def __init__(self, datasets: list[Dataset]):
        """
        Initialize the Analyzer with a list of datasets.

        Parameters:
        - datasets: List of Dataset objects to be analyzed.
        """
        self.datasets = datasets
        ray.init(runtime_env=runtime_env)
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        self.pbar = remote_tqdm.remote(total=self._sum_windows_count)

    def print_datasets(self):
        """Print the names of all datasets."""
        for dataset in self.datasets:
            print(dataset.name)

    @cached_property
    def _sum_series_count(self) -> int:
        """Calculate the total number of series across all datasets."""
        total_count = 0
        for dataset in self.datasets:
            total_count += dataset.hf_dataset.num_rows
        return total_count

    @cached_property
    def _sum_windows_count(self) -> int:
        """Calculate the total number of windows across all datasets."""
        total_count = 0
        for dataset in self.datasets:
            total_count += dataset.hf_dataset.num_rows * dataset.windows
        return total_count

    @property
    def freq_distribution_by_dataset(self):
        """Compute the frequency distribution by dataset."""
        freqs = [norm_freq_str(to_offset(dataset.freq).name)
                 for dataset in self.datasets]
        freq_counts = {freq: freqs.count(freq) for freq in set(freqs)}
        return freq_counts

    @property
    def freq_distribution_by_ts(self):
        """Compute the frequency distribution by time series count."""
        freq_ts_counts = defaultdict(lambda: 0)
        for dataset in self.datasets:
            freq_ts_counts[norm_freq_str(
                to_offset(dataset.freq).name)] += dataset.hf_dataset.num_rows
        return freq_ts_counts

    @property
    def freq_distribution_by_ts_length(self):
        """Compute the frequency distribution by time series length."""
        freq_dp_counts = defaultdict(lambda: 0)
        for dataset in self.datasets:
            freq_dp_counts[norm_freq_str(
                to_offset(dataset.freq).name)] += dataset.sum_series_length
        return freq_dp_counts

    @property
    def freq_distribution_by_window(self):
        """Compute the frequency distribution by window count."""
        freq_window_counts = defaultdict(lambda: 0)
        for dataset in self.datasets:
            freq_window_counts[norm_freq_str(
                to_offset(dataset.freq).name)] += dataset.hf_dataset.num_rows * dataset.windows
        return freq_window_counts

    def features_by_window(self, output_dir):
        """
        Create and persist features of each window for each dataset.

        Parameters:
        - output_dir: Directory where the features will be saved.
        """
        ray.get([process_dataset.remote(self, dataset, output_dir)
                for dataset in self.datasets])
        self.pbar.close.remote()

        all_datasets_df = []
        # Aggregate the characteristics for each dataset
        with tqdm(total=len(self.datasets), desc="Computing ts features for whole benchmark") as pbar:
            for dataset in self.datasets:
                pbar.set_description(f"Loading ts features | {dataset.name}")
                if str(dataset.term) == "Term.SHORT":
                    dataset_df_path = os.path.join(os.path.dirname(
                        output_dir), f"datasets/{dataset.name}/features.csv")
                else:
                    if "/" in dataset.name:
                        dataset_name, dataset_freq = dataset.name.split("/")
                        dataset_name = f"{dataset_name}:{dataset.term}/{dataset_freq}"
                        dataset_df_path = Path(os.path.join(os.path.dirname(
                            output_dir), f"datasets/{dataset_name}/features.csv"))
                    else:
                        dataset_df_path = Path(os.path.join(os.path.dirname(
                            output_dir), f"datasets/{dataset.name}:{dataset.term}/features.csv"))
                df = pd.read_csv(dataset_df_path)
                all_datasets_df.append(df)
                pbar.update(1)

            # Concatenate all dataset features and persist the analysis
            all_features_df = pd.concat(all_datasets_df, ignore_index=True)
            persist_analysis(all_features_df, output_dir)

        return None
