#!/usr/bin/env bash

file="$1"

tensorboard --logdir ./logs/lightning_logs &
python ./src/"$file"