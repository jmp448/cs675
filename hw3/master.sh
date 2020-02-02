#!/usr/bin/env bash

source python3-hw3/bin/activate

## Simple feed forward train
#python3 main.py train --data-dir data --log-file \
#        logs/ff-logs.csv --model-save models/ff.torch \
#        --model simple-ff --learning-rate 0.001 \
#        --min-iter 500 --max-iter 1000

## Simple feed forward parameter sweep
#python3 param_sweep.py train --data-dir data \
#        --model simple-ff --min-iter 500 \
#        --max-iter 1000
#
## Simple CNN train
#python3 main.py train --data-dir data \
#        --log-file logs/cnn-logs.csv \
#        --model-save models/cnn.torch \
#        --model simple-cnn \
#        --cnn-n1-channels 80 \
#        --min-iter 500 --max-iter 10000
#
## Simple CNN sweep
#python3 param_sweep.py train --data-dir data \
#        --model simple-cnn --min-iter 500 \
#        --max-iter 10000

# Best model train
#python3 main.py train --data-dir data \
#        --log-file logs/best-logs.csv \
#        --model-save models/best.torch \
#        --model best --min-iter 500 \
#        --max-iter 10000 --best-prob-drop=0.15 \
#        --batch-size=40 --best-n1-channels=40

# Best model sweep
#python3 param_sweep.py train --data-dir data \
#        --model best --min-iter 500 \
#        --max-iter 10000
