# GIFT-EVAL: A Benchmark for General Time Series Forecasting Model Evaluation

[![arXiv](https://img.shields.io/badge/GiftEval-2402.02592-b31b1b.svg)](https://arxiv.org/abs/2410.10393)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TrainTestDataset-FFD21E)](https://huggingface.co/datasets/Salesforce/GiftEval) 
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PretrainDataset-FFD21E)](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LeaderBoard-FFD21E)](https://huggingface.co/spaces/Salesforce/GIFT-Eval)

![gift eval main figure](artefacts/gifteval.png)

| Benchmark             | Freq. Range        | Num. of Domain | Pretraining data | Num. of var. | Pred. Len. | Benchmark Methods | Prob. Forecasting |
|-----------------------|--------------------|----------------|------------------|--------------|------------|-------------------|-------------------|
| Monash (Godahewa et al., 2021) | Secondly ~ Yearly | 7              | No               | Uni          | Short      | Stat./DL          | No                |
| TFB (Qiu et al., 2024)         | Minutely ~ Yearly | 6              | No               | Uni/Multi    | Short      | Stat./DL          | No                |
| LTSF (Zeng et al., 2022)       | Minutely ~ Weekly | 5              | No               | Multi        | Long       | Stat./DL          | No                |
| BasicTS+ (Shao et al., 2023)   | Minutely ~ Daily  | 3              | No               | Multi        | Short/Long | Stat./DL          | No                |
| GIFT-Eval (our work)           | Secondly ~ Yearly | 7              | Yes              | Uni/Multi    | Short/Long | Stat./DL/FM       | Yes               |

GIFT-Eval is a comprehensive benchmark designed to evaluate general time series forecasting models across diverse datasets, promoting the advancement of zero-shot forecasting capabilities in foundation models.

To facilitate the effective pretraining and evaluation of foundation models, we also provide a non-leaking pretraining dataset --> [GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain).

## Update Log

### 2025‑07‑24

- Corrected the Naive and Seasonal Naive scores to match the latest GIFT‑Eval notebooks. Most model rankings remain unchanged; only a few near the bottom shifted slightly (AutoETS and Timer each dropped two places now at 35th and 36th places respectively, while NBEATS moved up one now at 27th place).
### 2025-08-05

- Added new columns to the leaderboard: Organization, TestData Leakage, and MASE_Rank. TestData Leakage is a binary indicator specifying whether any test data was present in the training set. MASE_Rank reflects the model's ranking based on the MASE metric, aligned with the ranking scheme used for CRPS_Rank. These additions were made in response to multiple requests from independent groups seeking fairer comparisons. With these updates, the leaderboard now supports sorting by models that do not leak test data, and viewers can choose to rank models based on either MASE_Rank or CRPS_Rank, depending on their use case.

- Added new model type: Agentic to indicate submissions that use agentic system to generate the forecasts.

## Installation
1. Clone the repository and change the working directory to `GIFT_Eval`.
2. Create a conda environment:
```
python3 -m venv myenv
source myenv/bin/activate
```

3. Install required packages:

If you just want to explore the dataset, you can install the required dependencies as follows:
```
pip install -e .
```

If you want to run baselines, you can install the required dependencies as follows:
```
pip install -e .[baseline]
```
Note: The specific instructions for installing the [Moirai](notebooks/moirai.ipynb) and [Chronos](notebooks/chronos.ipynb) models are available in their relevant notebooks.

4. Get the train/test dataset from [huggingface](https://huggingface.co/datasets/Salesforce/GiftEval).

```
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir PATH_TO_SAVE
```

5. Set up the environment variables and add the path to the data:
```
echo "GIFT_EVAL=PATH_TO_SAVE" >> .env
```

## Getting Started

### Iterating the dataset

We provide a simple class, `Dataset` to load each dataset in our benchmark following the gluonts interface. It is highly recommended to use this class to split the data to train/val/test for compatibility with the evaluation framework and other baselines in the leaderboard. You don't have to stick to gluonts interface though as you can easily implement a wrapper class to load the data iterator in a different format than gluonts.

This class provides the following properties:

- `training_dataset`: The training dataset.
- `validation_dataset`: The validation dataset.
- `test_data`: The test dataset.

Please refer to the [dataset.ipynb](notebooks/dataset.ipynb) for an example of how to iterate the train/val/test splits of the dataset.
### Running baselines

We provide examples of how to run the statistical, deep learning, and foundation baselines in the [naive.ipynb](notebooks/naive.ipynb), [feedforward.ipynb](notebooks/feedforward.ipynb) and [moirai.ipynb](notebooks/moirai.ipynb) and [chronos.ipynb](notebooks/chronos.ipynb) notebooks. Each of these notebooks wrap models available in different libraries to help you get started. You can either follow these examples or implement your own wrapper class to iterate over the splits of the dataset as explained in the [dataset.ipynb](notebooks/dataset.ipynb) notebook.

Each of these notebooks will generate a csv file called `all_results.csv` under the `results/<MODEL_NAME>` folder containing the results for your model on the gift-eval benchmark. Regardless of the model you choose and how you run it, you can submit your results to the leaderboard by following the instructions in the [Submitting your results](#submitting-your-results) section.

### Sample output file
A sample output file is located at `results/naive/all_results.csv`.

The file contains the following columns:

- `dataset`: The name of the dataset configuration, e.g. `electricity/15T/short`.
- `model`: The name of the model, e.g. `naive`.
- A column for each evaluation metric used, e.g. `eval_metrics/MSE[mean]`, `eval_metrics/MSE[0.5]`, etc.
- `domain`: The domain of the dataset, e.g. `Web/CloudOps`.
- `num_variates`: The number of variates in the dataset, e.g. `1`.

The first column in the csv file is the dataset config name which is a combination of the prettified dataset name, frequency and the term (The sample notebooks, e.g. [naive.ipynb](notebooks/naive.ipynb), show how to get this name, please follow the same format to align with the leaderboard.):
```python
f"{dataset_name}/{freq}/{term}"
```

## Submitting your results

### Evaluation 

```python
res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=metrics,
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
```

We highly recommend you to evaluate your model using gluonts `evaluate_model` function as it is compatible with the evaluation framework and other baselines in the leaderboard. Please refer to the sample notebooks where we show its use with statistical, deep learning and foundation models for more details. However, if you decide to evaluate your model in a different way please follow the below conventions for compatibility with the rest of the baselines in our leaderboard. Specifically:

1. Aggregate results over all dimensions (following `axis=None`)
2. Do not count `nan` values in the target towards calculation (following  `mask_invalid_label=True`).
3. Make sure the prediction does not have `nan` values (following `allow_nan_forecast=False`).
   
### Submission
Submit your results to the leaderboard by creating a pull request that adds your results to the `results/<YOUR_MODEL_NAME>` folder. Your PR should contain only a folder with two files called `all_results.csv` and `config.json`. The `config.json` file should contain the following fields:
```json
{
    "model": "YOUR_MODEL_NAME",
    "model_type": "one of statistical, deep-learning, agentic, pretrained or fine-tuned",
    "model_dtype": "float32, etc."
    "model_link": "To your HF model link, e.g., https://huggingface.co/amazon/chronos-t5-small",
    "org": "YOUR_ORG_NAME",
    "testdata_leakage": "one of Yes or No" 
}
```

#### Field Descriptions

- **`model`**:  
  A short identifier for your model (e.g., `my_model_v1`).

- **`model_type`**:  
  Choose one of the following:
  - `statistical`: Traditional time series models such as ARIMA, ETS, etc.  
  - `deep-learning`: Neural network models trained from scratch.  
  - `agentic`: Multi-step systems that use agents or LLMs to reason, generate or select forecasts.  
  - `pretrained`: Foundation models trained once on large-scale data and applied as-is to each dataset.  
  - `fine-tuned`: Models that begin from a pretrained base but are further adapted (fine-tuned) individually on each dataset.
  
  > **Note:** The key difference between `pretrained` and `fine-tuned` is that fine-tuned models are adapted separately to each dataset using supervision, whereas pretrained models are used without per-dataset tuning.

- **`model_dtype`**:  
  The floating-point precision used in inference or training (e.g., `float32`, `bfloat16`, etc.).

- **`model_link`**:  
  A public link to your model (ideally on Hugging Face or another accessible hub).

- **`org`**:  
  The organization or team submitting the model.

- **`testdata_leakage`**:  
  Indicates whether the model has been trained on data that overlaps with our test datasets.

  > **Important:** We only consider leakage into the **test** split of the GIFT-Eval benchmark.  
  > If your training data includes any dataset that is part of our test corpus it must be labeled as `Yes`.  
  > Models trained solely on our provided [training split](https://huggingface.co/datasets/Salesforce/GiftEval) do **not** count as leaking, since our train splits are carefully constructed using earlier horizons that do not overlap with the test set.

The final `all_results.csv` file should contain `98` lines (one for each dataset configuration) and `15` columns: `4` for dataset, model, domain and num_variates and `11` for the evaluation metrics.

## Time Series Features Analysis

Add NUM_CPUS to your .env file to run the analysis in parallel.

```
echo "NUM_CPUS={N}" >> .env
```

To replicate the time series feature analysis in the paper, run the following command:

```
python -m cli.analysis datasets=all_datasets
```
This will run the analysis for all the datasets in the benchmark and generate two folders under `outputs/analysis/test`:
1. `datasets`: This folder contains the individual features for each dataset along with some some plots visualizing those features.
2. `all_datasets`: This folder contains the aggregated features for all the datasets along with some some plots visualizing those features.

Note: Expect the analysis to take long, we recommend running it on a large cpu cluster and setting the `NUM_CPUS` environment variable to the number of cores you have access to.

If you just want to try the analysis out you can run it with a few datasets by creating a new config file in the `cli/conf/analysis/datasets` folder. Follow the [`sample`](cli/conf/analysis/datasets/sample.yaml) file shared. 

```
python -m cli.analysis datasets=sample
```



## Citation
If you find this benchmark useful, please consider citing:

```
@article{aksu2024giftevalbenchmarkgeneraltime,
      title={GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation}, 
      author={Taha Aksu and Gerald Woo and Juncheng Liu and Xu Liu and Chenghao Liu and Silvio Savarese and Caiming Xiong and Doyen Sahoo},
      journal = {arxiv preprint arxiv:2410.10393},
      year={2024},
}
```

This repository is intended for research purposes only.
