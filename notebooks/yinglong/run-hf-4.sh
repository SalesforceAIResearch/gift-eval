model_name=qcw2333/YingLong_300m
future_token=4096

export CUDA_VISIBLE_DEVICES=0
python run-hf.py \
--batch_size 32 \
--model_name $model_name \
--future_token $future_token \
-s car_parts_with_missing \


export CUDA_VISIBLE_DEVICES=0
python run-hf.py \
--batch_size 1024 \
--model_name $model_name \
--future_token $future_token \
-s temperature_rain_with_missing -s m4_yearly  -s electricity/D -s restaurant -s kdd_cup_2018_with_missing/D -s covid_deaths -s M_DENSE/D -s  jena_weather/D -s saugeenday/D -s saugeenday/W -s m4_monthly \
-l LOOP_SEATTLE/5T -l kdd_cup_2018_with_missing/H -l SZ_TAXI/15T  -l ett1/15T -l bizitobs_l2c/5T -l bizitobs_l2c/H \
&

export CUDA_VISIBLE_DEVICES=1
python run-hf.py \
--batch_size 1024 \
--model_name $model_name \
--future_token $future_token \
-s bitbrains_fast_storage/H  -s m4_daily -s electricity/W -s hierarchical_sales/W -s m4_weekly   -s  ett2/W -s us_births/W -s  us_births/M \
-l  bitbrains_rnd/5T -l electricity/H -l bizitobs_service  -l  jena_weather/H -l ett2/15T  \
&

export CUDA_VISIBLE_DEVICES=2
python run-hf.py \
--batch_size 1024 \
--model_name $model_name \
--future_token $future_token \
-s hospital -s LOOP_SEATTLE/D -s m4_hourly -s ett1/D -s ett2/D -s ett1/W -s saugeenday/M \
-l bitbrains_fast_storage/5T -l solar/10T -l M_DENSE/H -l ett1/H -l  bizitobs_application \
&

export CUDA_VISIBLE_DEVICES=3
python run-hf.py \
--batch_size 1024 \
--model_name $model_name \
--future_token $future_token \
-s m4_quarterly -s bitbrains_rnd/H -s hierarchical_sales/D  -s SZ_TAXI/H -s solar/D  -s solar/W   -s us_births/D \
-l electricity/15T -l LOOP_SEATTLE/H -l solar/H -l jena_weather/10T -l ett2/H \
&