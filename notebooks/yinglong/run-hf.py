import argparse
import os
import torch
import json


import logging
from transformers import AutoModelForCausalLM

class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)




from dotenv import load_dotenv

from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)


from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm
from  einops import rearrange
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset



@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


class YingLongPredictor:
    def __init__(
        self,
        model,
        prediction_length: int,
        num_samples=20,
        future_token = 4096,
        *args,
        **kwargs,
    ):
        print("prediction_length:", prediction_length)
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.model = model
        self.future_token = future_token

        
        
    def model_predict(self, 
                      context,
                      prediction_length,
                      future_token,
                      scaling = 400,
                      max_length = 4096*16,
                      *args,**predict_kwargs):

        
        context = [torch.nan_to_num(x[-max_length:].to(gpu_device),nan=torch.nanmean(x[-max_length:].to(gpu_device))) for x in  context]
       
        length = max([len(x) for x in context])
        context = [x[-length:] if len(x) >= length else torch.cat((torch.ones(length- x.shape[-1]).to(x.device)*torch.mean(x),x)) for x in  context]
        x = torch.stack(context,dim = 0)

        scale_factor = 1
        with torch.no_grad():
            B, _, = x.shape


            logits = 0

                        
            historys = [512,1024,2048,4096]
            if future_token<1000:
                future_token = (prediction_length//32 + 1)*32

            used = 0
            for history in historys:
                if used == 0 or history <= x.shape[-1]:
                    used += 2
                else:
                    continue
                x_train = torch.cat((x.bfloat16(),-x.bfloat16()),dim = 0)
                x_train = x_train[...,-history:].bfloat16()

                if x_train.shape[-1] % self.model.patch_size != 0:
                    shape = (x_train.shape[0],self.model.patch_size -  x.shape[-1] % self.model.patch_size)
                    x_train = torch.cat((torch.ones(shape).to(x_train.device)*x_train.mean(dim = -1,keepdims = True),x_train),dim = -1)
                    x_train = x_train.bfloat16()

                
                logits_all = model(idx = x_train, future_token = future_token)
                    
                logits_all = rearrange(logits_all, '(t b) l c d -> b (l c) d t', t = 2)
                logits += logits_all[...,0]  - logits_all[...,1].flip(dims = [-1])
              
            logits = logits / used

            sampleHolder = rearrange(logits, 'b l c ->b c l').float().contiguous().cpu().detach()[:,:,:prediction_length]
            return torch.nan_to_num(sampleHolder)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
      
        predict_kwargs=  {"num_samples": self.num_samples}
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    forecast_outputs.append(
                         self.model_predict(context,
                              prediction_length = self.prediction_length,
                                future_token = self.future_token,
                           **predict_kwargs,
                             ).numpy()
                    )
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(SampleForecast(samples=item, start_date=forecast_start_date))
          

        return forecasts


if __name__ == '__main__':
    
    


    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
    parser.add_argument('--model_name', type=str,  default='qcw1314/YingLong_6m')
    parser.add_argument('--seed', type=int, default=3407, help='model size')
    parser.add_argument('--future_token', type=int,default=3072)
    parser.add_argument('--output_dir', type=str,default='results_hf_0')
    parser.add_argument('-l', '--long_tasks',default = [], action='append')
    parser.add_argument('-s', '--short_tasks',default = [],action='append')
    args = parser.parse_args()
    




    # Load environment variables
    load_dotenv()
  
    short_datasets = args.short_tasks
    med_long_datasets = args.long_tasks
    all_datasets = short_datasets + med_long_datasets
    all_datasets = list(set(all_datasets))
    dataset_properties_map = json.load(open("dataset_properties.json"))
    
    
    
        # Instantiate the metrics
    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]

  
    gpu_device = 'cuda:0'
  
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)#,torch_dtype=torch.bfloat16).cuda()

    model = model.to(gpu_device).bfloat16()

    model.eval()
    
    
    
    
    model_name = f"{args.model_name.split('/')[-1]}-{args.future_token}-4096"
    
    output_dir = f"{args.output_dir}/{model_name}"
    if not os.path.isdir(output_dir):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)


    # Define the path for the CSV file
    csv_file_path = os.path.join(output_dir, "all_results.csv")
    
    pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
    }
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "eval_metrics/MSE[mean]",
                    "eval_metrics/MSE[0.5]",
                    "eval_metrics/MAE[0.5]",
                    "eval_metrics/MASE[0.5]",
                    "eval_metrics/MAPE[0.5]",
                    "eval_metrics/sMAPE[0.5]",
                    "eval_metrics/MSIS",
                    "eval_metrics/RMSE[mean]",
                    "eval_metrics/NRMSE[mean]",
                    "eval_metrics/ND[0.5]",
                    "eval_metrics/mean_weighted_sum_quantile_loss",
                    "domain",
                    "num_variates",
                ]
            )
            
            
            
    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        terms = ["short", "medium", "long"]
        for term in terms:
            if (
                term == "medium" or term == "long"
            ) and ds_name not in med_long_datasets:
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            # Initialize the dataset
            to_univariate = (
                False
                if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                else True
            )
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)
            print(f"Dataset size: {len(dataset.test_data)}")
            predictor = YingLongPredictor(
                model = model,
                prediction_length=dataset.prediction_length,
                future_token = args.future_token,
                device_map="cuda",
            )
            # Measure the time taken for evaluation
            res = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=1024,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )

            # Append the results to the CSV file
            with open(csv_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        ds_config,
                        model_name,
                        res["MSE[mean]"][0],
                        res["MSE[0.5]"][0],
                        res["MAE[0.5]"][0],
                        res["MASE[0.5]"][0],
                        res["MAPE[0.5]"][0],
                        res["sMAPE[0.5]"][0],
                        res["MSIS"][0],
                        res["RMSE[mean]"][0],
                        res["NRMSE[mean]"][0],
                        res["ND[0.5]"][0],
                        res["mean_weighted_sum_quantile_loss"][0],
                        dataset_properties_map[ds_key]["domain"],
                        dataset_properties_map[ds_key]["num_variates"],
                    ]
                )

            print(f"Results for {ds_name} have been written to {csv_file_path}")