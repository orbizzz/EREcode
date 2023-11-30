# cd FEDformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi


for preLen in 24 36 48 60
do
python -u run.py \
 --is_training 1 \
 --data_path price_total.csv \
 --task_id price \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 9 \
 --dec_in 9 \
 --c_out 9 \
 --des 'Exp' \
 --itr 1 >../logs/LongForecasting/FEDformer_price_$preLen.log

python -u run.py \
 --is_training 1 \
 --data_path yieldcurve.csv \
 --task_id yield \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 10 \
 --dec_in 10 \
 --c_out 10 \
 --des 'Exp' \
 --itr 1 >../logs/LongForecasting/FEDformer_yield_$preLen.log
done

# cd ..