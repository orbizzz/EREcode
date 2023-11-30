# cd Pyraformer
if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

# price_total
python long_range_main.py  -window_size [2,2,2] -data_path price_total.csv -data price \
-input_size 24 -predict_step 24 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_price_24.log
python long_range_main.py  -window_size [2,2,2] -data_path price_total.csv -data price \
-input_size 24 -predict_step 36 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_price_36.log
python long_range_main.py  -window_size [2,2,2] -data_path price_total.csv -data price \
-input_size 24 -predict_step 48 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_price_48.log
python long_range_main.py  -window_size [2,2,2] -data_path price_total.csv -data price \
-input_size 24 -predict_step 60 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_price_60.log

# yieldcurve
python long_range_main.py  -window_size [2,2,2] -data_path yieldcurve.csv -data yield \
-input_size 24 -predict_step 24 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_yield_24.log
python long_range_main.py  -window_size [2,2,2] -data_path yieldcurve.csv -data yield \
-input_size 24 -predict_step 36 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_yield_36.log
python long_range_main.py  -window_size [2,2,2] -data_path yieldcurve.csv -data yield \
-input_size 24 -predict_step 48 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_yield_48.log
python long_range_main.py  -window_size [2,2,2] -data_path yieldcurve.csv -data yield \
-input_size 24 -predict_step 60 -n_head 6 -lr 0.00001 -d_model 256 >../logs/LongForecasting/Pyraformer_yield_60.log

