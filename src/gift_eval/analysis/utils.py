import matplotlib.pyplot as plt
import numpy as np
import os


def normalize_data(mean, std):
    """
    Normalize the data using Min-Max Scaling to bring the mean and standard deviation between 0 and 1.

    Parameters:
    - mean: The mean values of the data.
    - std: The standard deviation values of the data.

    Returns:
    - Tuple of normalized mean and standard deviation.
    """
    # Apply Min-Max Scaling to normalize the data between 0 and 1
    mean_normalized = (mean - mean.min()) / (mean.max() - mean.min())
    # Normalize std by the same range for consistency
    std_normalized = std / (mean.max() - mean.min())
    return mean_normalized, std_normalized


def plot_radar_chart(df, output_dir):
    """
    Plot a radar chart for the given DataFrame and save it to the specified output directory.

    Parameters:
    - df: DataFrame containing the data to plot.
    - output_dir: Directory where the radar chart will be saved.
    """
    # Calculate mean and std across rows for each column
    mean = df.mean()
    std = df.std()

    # Normalize data
    mean_normalized, std_normalized = normalize_data(mean, std)

    # Number of variables we're plotting.
    num_vars = len(df.columns)

    # Split the circle into even parts and save angles so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Complete the loop for the plot
    angles += angles[:1]
    mean_normalized = mean_normalized.tolist() + mean_normalized.tolist()[:1]
    std_normalized = std_normalized.tolist() + std_normalized.tolist()[:1]

    # Draw the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, mean_normalized, color='red', alpha=0.25)
    ax.plot(angles, mean_normalized, color='red', label='Mean')

    # Draw error bars
    for angle, mean, error in zip(angles, mean_normalized, std_normalized):
        ax.errorbar(angle, mean, yerr=error, color='black', capsize=3)

    ax.set_ylim(0, 1)

    # Labels for each feature
    labels = df.columns.tolist()
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)

    # Title and legend
    plt.title('Time Series Benchmark Features (Normalized)',
              size=15, color='red', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'character_radar.png'))
    plt.close()


def persist_analysis(features_df, output_dir):
    """
    Persist the analysis results by saving the features DataFrame and its description to CSV files.

    Parameters:
    - features_df: DataFrame containing the features to be saved.
    - output_dir: Directory where the CSV files will be saved.
    """
    features_df.to_csv(f"{output_dir}/features.csv")
    features_df.describe().to_csv(f"{output_dir}/features_description.csv")

    # Create histograms of each characteristic
    for column in features_df.columns:
        plot_feature_histogram(features_df, column, output_dir)
    # Create a ring plot showing the mean and std of some key features, trend and seasonality should be there for sure.
    plot_radar_chart(features_df, output_dir)


def plot_histogram(freq_distribution_dict, name, output_dir):
    """
    Plot the histogram for the given frequency distribution dictionary and save it to the output directory.

    Parameters:
    - freq_distribution_dict: Dictionary with frequency strings as keys and counts as values.
    - name: Name of the frequency distribution.
    - output_dir: Directory where the histogram will be saved.
    """
    fig, ax = plt.subplots()
    ax.bar(freq_distribution_dict.keys(), freq_distribution_dict.values())
    ax.set_xlabel(f"Frequency of {name}")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {name} frequencies")
    plt.savefig(f"{output_dir}/{name}_frequency_distribution.png")
    plt.close()


def plot_feature_histogram(dataframe, column_name, output_directory):
    """
    Plot a histogram for the given column in the DataFrame and save it to the specified output directory.

    Parameters:
    - dataframe: The DataFrame containing the data.
    - column_name: The column name for which to plot the histogram.
    - output_directory: The directory where the plot will be saved.
    """
    # Check if the column exists in the dataframe
    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' does not exist in the dataframe.")

    # Drop rows with NA values in the specified column
    clean_data = dataframe[column_name].dropna()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(clean_data, bins=10, edgecolor='black')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    # Save the plot
    plot_filename = os.path.join(
        output_directory, f"{column_name}_histogram.png")
    plt.savefig(plot_filename)
    plt.close()
