import json
from os.path import join, dirname, split
import os
from datetime import datetime
from pprint import pprint
import shutil
import traceback

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np
import lightning as L
import torch.nn.functional as F
from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import BackboneFinetuning
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from ray.train import Checkpoint
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
# import tensorflow_addons as tfa

# from ray import tune
import ray
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import air, tune
from ray.tune.callback import Callback
from ray.tune.experiment.trial import Trial
from torchmetrics.classification import MulticlassConfusionMatrix
# import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.constants import ROOT_DIR, PADDING_SEC
from model import Wav2Vec2Classifier, Wav2Vec2CnnClassifier
from metrics.confusion_matrix import CreateConfMatrix
from src.train.train_func import train_func
from src.train.tune.tune_controller import start_tuning


def main():
    start_tuning()  # запустить тюнинг
    train_func(
        {
            'lr': 1e-3,
            'batch_size': 64,
            'num_workers': 1,
            'alpha': 0 * 0.25,
            "gamma": 0 * 2.0,
            "reduction": "mean",
            "from_logits": False,
        }

        # {
        #     'layer_1_size': 1024,
        #  'layer_2_size': 64,
        #  'lr': 0.008769148370041635,
        #  'batch_size': 64, 'conv_h_count': 16,
        #  'conv_w_count': 2, 'num_workers': 1,
        #  'alpha': 1, "gamma": 0.25,
        #  "reduction": "none",
        #  "from_logits": False,
        #  }

        # {
        #     'batch_size': 16,
        #     'conv_h_count': 8,
        #     'conv_w_count': 0,
        #     'layer_1_size': 512,
        #     'layer_2_size': 256,
        #     'lr': 0.0035923573027211784,
        #     'num_workers': 1
        # }
        # {'batch_size': 8, 'conv_h_count': 2, 'conv_w_count': 0,
        #  'layer_1_size': 1024, 'layer_2_size': 1024,
        #  'lr': 0.006854079146121017, 'num_workers': 1}
    )




if __name__ == '__main__':
    # check_tuning_res('tune_analyzing_results_20240416-141925')
    main()

'''
(0.8125, {'train_loop_config': {'batch_size': 16, 'conv_h_count': 8, 'conv_w_count': 0, 'layer_1_size': 512, 'layer_2_size': 256, 'lr': 0.0035923573027211784, 'num_workers': 1}})
(0.7954545617103577, {'train_loop_config': {'batch_size': 8, 'conv_h_count': 2, 'conv_w_count': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.006854079146121017, 'num_workers': 1}})
(0.7954545617103577, {'train_loop_config': {'batch_size': 8, 'conv_h_count': 0, 'conv_w_count': 0, 'layer_1_size': 2048, 'layer_2_size': 256, 'lr': 0.00012928256749833867, 'num_workers': 1}})
(0.7727272510528564, {'train_loop_config': {'batch_size': 8, 'conv_h_count': 4, 'conv_w_count': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.003983555045340102, 'num_workers': 1}})
(0.7613636255264282, {'train_loop_config': {'batch_size': 4, 'conv_h_count': 8, 'conv_w_count': 2, 'layer_1_size': 1024, 'layer_2_size': 64, 'lr': 0.0007337747082560383, 'num_workers': 1}})
(0.7272727489471436, {'train_loop_config': {'batch_size': 4, 'conv_h_count': 16, 'conv_w_count': 0, 'layer_1_size': 256, 'layer_2_size': 128, 'lr': 0.0013735981461098664, 'num_workers': 1}})

{'acc': 0.796875, 'epoch': 9, 'id': '55119_00068', 'layer_1_size': 1024, 'layer_2_size': 64, 'lr': 0.008769148370041635, 'batch_size': 64, 'conv_h_count': 16, 'conv_w_count': 2, 'num_workers': 1}
 {'acc': 0.796875, 'epoch': 9, 'id': '55119_00473', 'layer_1_size': 64, 'layer_2_size': 64, 'lr': 0.0036585834709462668, 'batch_size': 32, 'conv_h_count': 8, 'conv_w_count': 16, 'num_workers': 1}
{'acc': 0.796875, 'epoch': 9, 'id': '55119_00484', 'layer_1_size': 128, 'layer_2_size': 64, 'lr': 0.002930486390263172, 'batch_size': 32, 'conv_h_count': 8, 'conv_w_count': 0, 'num_workers': 1}
 {'acc': 0.7875000238418579, 'epoch': 9, 'id': '55119_00315', 'layer_1_size': 64, 'layer_2_size': 64, 'lr': 0.005135463347355345, 'batch_size': 16, 'conv_h_count': 8, 'conv_w_count': 4, 'num_workers': 1}
 {'acc': 0.78125, 'epoch': 5, 'id': '55119_00045', 'layer_1_size': 1024, 'layer_2_size': 32, 'lr': 0.004106611253125662, 'batch_size': 32, 'conv_h_count': 2, 'conv_w_count': 1, 'num_workers': 1}
 {'acc': 0.78125, 'epoch': 8, 'id': '55119_00200', 'layer_1_size': 128, 'layer_2_size': 256, 'lr': 0.003912079786909736, 'batch_size': 64, 'conv_h_count': 2, 'conv_w_count': 1, 'num_workers': 1}
 {'acc': 0.78125, 'epoch': 9, 'id': '55119_00211', 'layer_1_size': 512, 'layer_2_size': 128, 'lr': 0.0038767867780932245, 'batch_size': 32, 'conv_h_count': 2, 'conv_w_count': 0, 'num_workers': 1}
{'acc': 0.75, 'epoch': 6, 'id': '55119_00089', 'layer_1_size': 1024, 'layer_2_size': 128, 'lr': 0.01002987738490128, 'batch_size': 32, 'conv_h_count': 2, 'conv_w_count': 4, 'num_workers': 1}
 {'acc': 0.75, 'epoch': 9, 'id': '55119_00348', 'layer_1_size': 1024, 'layer_2_size': 32, 'lr': 0.0024426879784382126, 'batch_size': 16, 'conv_h_count': 8, 'conv_w_count': 8, 'num_workers': 1}
 {'acc': 0.75, 'epoch': 5, 'id': '55119_00482', 'layer_1_size': 2048, 'layer_2_size': 32, 'lr': 0.003605858112253073, 'batch_size': 16, 'conv_h_count': 2, 'conv_w_count': 0, 'num_workers': 1}

'''
