from datetime import datetime
from os.path import join

import lightning as L
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from ray.train.lightgbm import RayTrainReportCallback
from ray.train.lightning import RayLightningEnvironment, RayDDPStrategy, prepare_trainer

from src.config.constants import ROOT_DIR, PADDING_SEC
from src.data_controller.load_data import load_data
from src.train.lighting_model import LitModule


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
        learning_rate=config["lr"],
        conv_learning_rate=1e-3,

        # conv_h_count=config["conv_h_count"],
        # conv_w_count=config["conv_w_count"],
        padding_sec=PADDING_SEC,
        # layer_1_size=config["layer_1_size"],
        # layer_2_size=config["layer_2_size"],
        lables=dataset.emotions,
        dataset=dataset,
        gamma=config["gamma"],
        config=config,
    )
    model = LitModule(
        **lit_model_params
    )

    # path = join(ROOT_DIR, "weights", "checkpoints", "20240416-220411", "")

    path = join(
        ROOT_DIR, "tmp", "tune", "tune_analyzing_results_20240416-141925",
        "TorchTrainer_55119_00068_68_batch_size=64,conv_h_count=16,conv_w_count=2,layer_1_size=1024,layer_2_size=64,lr=0.0088_2024-04-16_14-19-27",
        "checkpoint_000009", "checkpoint.ckpt")
    # path = join(
    #     ROOT_DIR, "weights", "checkpoints", "20240416-220411",
    #     "classifier_20240416-220411_epoch=16.ckpt")
    # model = LitModule.load_from_checkpoint(path, **lit_model_params)

    # disable randomness, dropout, etc...
    # model.eval()
    # model.model.classifier.load_state_dict(torch.load(path))
    # model.model.classifier.eval()

    # model.model.load_state_dict(torch.load(path))

    # checkpoint = session.get_checkpoint()

    # if checkpoint:
    #     checkpoint_state = checkpoint.to_dict()
    #     start_epoch = checkpoint_state["epoch"]
    #     net.load_state_dict(checkpoint_state["net_state_dict"])
    #     optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    # else:
    #     start_epoch = 0

    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    # model.to(device)

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
