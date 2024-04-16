#!/usr/bin/bash

file="$1"

tensorboard --logdir ./logs/lightning_logs &
#jupyter lab  --allow-root  --ip=0.0.0.0 --NotebookApp.token=''  &
 python ./src/"$file"
jupyter lab  --allow-root  --ip=0.0.0.0 --NotebookApp.token=''
