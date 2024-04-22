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

from config.constants import ROOT_DIR, PADDING_SEC
from data_controller.load_data import load_data
from train.lighting_model import LitModule


def train_func(config, tuning=False):
    train_dataloader, val_dataloader, dataset = load_data(
        bath_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    checkpoint_callback = ModelCheckpoint(
        **dict(
            dirpath=join(ROOT_DIR, "weights", 'checkpoints', datetime.now().strftime("%Y%m%d-%H%M%S")),
            filename="classifier_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_{epoch:02d}",
            monitor="val/acc",
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
        num_classes=len(dataset.emotions),
        dataset=dataset,
        config=config,
    )
    model = LitModule(
        **lit_model_params
    )

    path = join(
        ROOT_DIR, "tmp", "tune", "tune_analyzing_results_20240416-141925",
        "TorchTrainer_55119_00068_68_batch_size=64,conv_h_count=16,conv_w_count=2,layer_1_size=1024,layer_2_size=64,lr=0.0088_2024-04-16_14-19-27",
        "checkpoint_000009", "checkpoint.ckpt")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = L.Trainer(**(dict(accelerator='gpu',
                                devices=1,
                                max_epochs=1000,
                                callbacks=[
                                              checkpoint_callback,
                                              lr_monitor,
                                          ] + ([RayTrainReportCallback()] if tuning else []),
                                default_root_dir=join(ROOT_DIR, 'logs')
                                ) | (dict(
        plugins=[RayLightningEnvironment()],
        strategy=RayDDPStrategy(find_unused_parameters=True),

    ) if tuning else dict(strategy='auto'))))
    if tuning:
        trainer = prepare_trainer(trainer)
    return trainer.fit(model, train_dataloader, val_dataloader)  # , ckpt_path=path
