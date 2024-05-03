#!/bin/bash

file="$1"

#  /workspace/NN/tmp/ray/session/session_2024-05-02_15-08-57_268970_8/artifacts/2024-05-02_15-08-59/tune_20240502-150857/driver_artifacts
# name1:/workspace/NN/logs/lightning_logs__test,name2:/workspace/NN/tmp/ray/session/session_2024-05-02_15-08-57_268970_8/artifacts/2024-05-02_15-08-59/tune_20240502-150857/driver_artifacts
tensorboard --logdir_spec $(python ./src/utils/get_tensorboard_logs_files.py)   --host 0.0.0.0 &
#jupyter lab  --allow-root  --ip=0.0.0.0 --NotebookApp.token=''  &
python ./src/$*
jupyter lab  --allow-root  --ip=0.0.0.0 --NotebookApp.token=''
#python ./src/utils/get_tensorboard_logs_files.py