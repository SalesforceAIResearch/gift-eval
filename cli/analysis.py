
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from gift_eval.analysis.utils import plot_histogram


@hydra.main(version_base="1.3", config_path="conf/analysis", config_name="default")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    analyzer = instantiate(cfg.analyzer, _convert_="all")
    analyzer.print_datasets()

    print(analyzer.freq_distribution_by_dataset)
    print(analyzer.freq_distribution_by_ts)
    print(analyzer.freq_distribution_by_ts_length)
    print(analyzer.freq_distribution_by_window)

    # plot a histogram of all three frequncy distributions and save it to output_dir
    plot_histogram(analyzer.freq_distribution_by_dataset,
                   "dataset", output_dir)
    plot_histogram(analyzer.freq_distribution_by_ts, "time series", output_dir)
    plot_histogram(analyzer.freq_distribution_by_ts_length,
                   "ts length", output_dir)
    plot_histogram(analyzer.freq_distribution_by_window, "window", output_dir)

    analyzer.features_by_window(output_dir)


if __name__ == "__main__":
    main()
