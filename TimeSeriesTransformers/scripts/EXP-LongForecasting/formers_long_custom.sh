# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

for model_name in Autoformer Informer Transformer
do 
for pred_len in 24 36 48 60
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path price_total.csv \
    --model_id price_total_36_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 36 \
    --label_len 18 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 9 \
    --dec_in 9 \
    --c_out 9 \
    --des 'Exp' \
    --itr 1 >logs/LongForecasting/$model_name'_price_total_'$pred_len.log
done
done

for model_name in Autoformer Informer Transformer
do 
for pred_len in 24 36 48 60
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path yieldcurve.csv \
    --model_id yieldcurve_36_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 36 \
    --label_len 18 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 10 \
    --dec_in 10 \
    --c_out 10 \
    --des 'Exp' \
    --itr 1 >logs/LongForecasting/$model_name'_yieldcurve_'$pred_len.log
done
done
