#!/bin/bash

cd FEDformer
sh ./scripts/LongForecasting.sh
sh ./scripts/longforecasting_custom.sh
cd ..
cd Pyraformer
echo Pyraformer
sh ./scripts/LongForecasting.sh
sh ./scripts/longforecasting_custom.sh
cd ..
echo Former
sh ./scripts/EXP-LongForecasting/Formers_Long.sh
sh ./scripts/EXP-LongForecasting/formers_long_custom.sh
echo Stat
sh ./scripts/EXP-LongForecasting/Stat_Long.sh
sh ./scripts/EXP-LongForecasting/stat_long_custom.sh
