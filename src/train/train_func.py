from datetime import datetime
from os.path import join

import lightning as L
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from lightning.pytorch.loggers import TensorBoardLogger

from config.constants import ROOT_DIR, PADDING_SEC
from data_controller.load_data import load_data
from train.lighting_model import LitModule
from data_controller.emotion_dataset import EmotionSpectrogramDataset, EmotionDataset


def dataset_choice(config):
    if config['learn_params'].get('padding_sec'):
        return EmotionDataset
    elif config['learn_params'].get('spectrogram_size'):
        return EmotionSpectrogramDataset


def train_func(
        config,
        # tuning=False,
        # enable_tune_features=False,
        # saved_checkpoint: str | None = None
):
    train_dataloader, val_dataloader, dataset = load_data(
        bath_size=config['learn_params']["batch_size"],
        num_workers=config["load_dataset_workers_num"],
        dataset_class=dataset_choice(config),
        padding_sec=config['learn_params'].get('padding_sec', 0)
    )

    tune = tuning = config.get("tune", False)
    enable_tune_features = tune and config['tune'].get("enable_tune_features", False)
    saved_checkpoint = (config['saving_data_params'].get('start_from_saved_checkpoint_path', None)
                        and join(*config['saving_data_params']['start_from_saved_checkpoint_path']))

    checkpoint_callback = ModelCheckpoint(
        **dict(
            dirpath=join(*config['saving_data_params']['saved_checkpoints_path']),
            filename=''.join(config['saving_data_params']['saved_checkpoints_filename']),
            monitor="val/em_acc",
            mode='max',  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)

        ) | (
              dict(every_n_epochs=30)
              if tuning else
              dict(
                  save_top_k=1,  # Save top k checkpoints based on the monitored metric
                  save_last=True,  # Save the last checkpoint at the end of training))
              )))
    # print(ModelCheckpoint.dirpath)

    lit_model_params = dict(
        emotions_count=len(dataset.emotions),
        states_count=len(dataset.states),
        dataset=dataset,
        config=config,
    )
    model = LitModule(
        **lit_model_params
    )

    sub_dir = None
    if config['saving_data_params'].get('tensorboard_lr_monitors_logs_sub_dir'):
        sub_dir = ''.join(config['saving_data_params']['tensorboard_lr_monitors_logs_sub_dir'])
    a_tensorboard_logger = TensorBoardLogger(
        save_dir=join(*(config['saving_data_params']['tensorboard_lr_monitors_logs_path'][:-1])),
        version="".join(config['saving_data_params']['tensorboard_lr_monitors_logs_name']),
        name=''.join(config['saving_data_params']['tensorboard_lr_monitors_logs_path'][-1]),
        sub_dir=sub_dir
    )
    # B
    # WandbLogger(save_dir=os.getcwd())

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = L.Trainer(**(dict(accelerator='gpu',
                                devices=1,
                                logger=a_tensorboard_logger,
                                max_epochs=config['learn_params']['max_epoch'],
                                callbacks=[
                                              checkpoint_callback,
                                              lr_monitor,
                                          ] + ([RayTrainReportCallback()] if tuning else []),
                                default_root_dir=join(
                                    *config['saving_data_params']['tensorboard_lr_monitors_logs_path']
                                )
                                ) | (dict(
        plugins=[RayLightningEnvironment()],
        strategy=RayDDPStrategy(find_unused_parameters=True),

    ) if tuning else dict(strategy='auto'))))
    if tuning and enable_tune_features:
        trainer = prepare_trainer(trainer)
    return trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=saved_checkpoint)  # , ckpt_path=path
