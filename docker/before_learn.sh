#!/bin/bash

file="$1"

tensorboard --logdir ./logs/lightning_logs  --host 0.0.0.0 &
jupyter lab  --allow-root  --ip=0.0.0.0 --NotebookApp.token=''  &
python ./src/$*
jupyter lab  --allow-root  --ip=0.0.0.0 --NotebookApp.token=''
