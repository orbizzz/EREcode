if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

for model_name in Naive
  do
  for pred_len in 24 36 48 60
    do
      python -u run_stat.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path price_total.csv \
          --model_id price_total_36'_'$pred_len \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 36 \
          --label_len 18 \
          --pred_len $pred_len \
          --des 'Exp' \
          --itr 1 >logs/LongForecasting/$model_name'_price_total_'$pred_len.log
  done
done

for model_name in Naive
do
for pred_len in 24 36 48 60
do
      python -u run_stat.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path yieldcurve.csv \
          --model_id yieldcurve_36'_'$pred_len \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 36 \
          --label_len 18 \
          --pred_len $pred_len \
          --des 'Exp' \
          --itr 1 >logs/LongForecasting/$model_name'_yieldcurve_'$pred_len.log
done
done