# GIFT-EVAL: A Benchmark for General Time Series Forecasting Model Evaluation

[Paper](https://arxiv.org/abs/2410.10393) | [Blog Post]() | [Dataset]() | [Leaderboard]()

![gift eval main figure](artefacts/gifteval.png)

GIFT-Eval is a comprehensive benchmark designed to evaluate general time series forecasting models across diverse datasets, promoting the advancement of zero-shot forecasting capabilities in foundation models.
## Installation
1. Clone the repository and change the working directory to `GIFT_Eval`.
2. Create a conda environment:
```
virtualenv venv
. venv/bin/activate
```

3. Install required packages:

If you just want to explore the dataset, you can install the required dependencies as follows:
```
pip install -e .
```

If you want to run baselines, you can install the required dependencies as follows:
```
pip install -e .[notebook]
```

4. Get the train/test dataset from [huggingface]().

5. Set up the environment variables:
Create a `.env` file in the root directory of the project and add the following:
```
GIFT_EVAL=/path/to/gift_eval/data # Required
```
Replace the paths with the appropriate locations on your system.

## Getting Started

### Iterating the dataset

We provide a simple class, `Dataset` to load each dataset in our benchmark following the gluonts interface. It is highly recommended to use this class to split the data to train/val/test for compatibility with the evaluation framework and other baselines in the leaderboard. You don't have to stick to gluonts interface though as you can easily implement a wrapper class to load the data iterator in a different format than gluonts.

This class provides the following properties:

- `training_dataset`: The training dataset.
- `validation_dataset`: The validation dataset.
- `test_data`: The test dataset.

Please refer to the [dataset.ipynb](notebooks/dataset.ipynb) for an example of how to iterate the train/val/test splits of the dataset.
### Running baselines

We provide examples of how to run the statistical, deep learning, and foundation baselines in the [naive.ipynb](notebooks/naive.ipynb), [feedforward.ipynb](notebooks/feedforward.ipynb) and [moirai.ipynb](notebooks/moirai.ipynb) notebooks. Each of these notebooks wrap models available in different libraries to help you get started. You can either follow these examples or implement your own wrapper class to iterate over the splits of the dataset as explained in the [dataset.ipynb](notebooks/dataset.ipynb) notebook.

Each of these notebooks will generate a csv file called `all_results.csv` under the `results/<MODEL_NAME>` folder containing the results for your model on the gift-eval benchmark. Regardless of the model you choose and how you run it, you can submit your results to the leaderboard by following the instructions in the [Submitting your results](#submitting-your-results) section.

### Sample output file
A sample output file is located at `results/naive/all_results.csv`.

The file contains the following columns:

- `dataset`: The name of the dataset configuration, e.g. `electricity/15T/short`.
- `model`: The name of the model, e.g. `naive`.
- A column for each evaluation metric used, e.g. `eval_metrics/MSE[mean]`, `eval_metrics/MSE[0.5]`, etc.
- `domain`: The domain of the dataset, e.g. `Web/CloudOps`.
- `num_variates`: The number of variates in the dataset, e.g. `1`.

The first column in the csv file is the dataset config name which is a combination of the dataset name, frequency and the term:
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
    "model_type": "one of statistical, deep-learning, or pretrained",
    "model_dtype": "float32, etc."
}
```

The final `all_results.csv` file should contain `98` lines (one for each dataset configuration) and `15` columns: `4` for dataset, model, domain and num_variates and `11` for the evaluation metrics.

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
