from datetime import datetime
from os.path import join
import traceback

import torch
from pytorch_lightning.callbacks import Checkpoint
from ray import tune
from ray.air import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from config.constants import ROOT_DIR
from train.callbacks.delete_checkpoint import DeleteCallback
from train.train_func import train_func
from config.constants import ROOT_DIR, PADDING_SEC


def tune_wapper_of_train_func(config):
    torch.cuda.empty_cache()
    tune.utils.wait_for_gpu()
    try:
        result = train_func(
            config,
            tuning=True,
        )
    except Exception as e:
        print("some error win train", e, traceback.format_exc())
        torch.cuda.empty_cache()
        tune.utils.wait_for_gpu()
        raise e from e


def dict_chain(sp_config, config, __deep=0):
    if isinstance(sp_config, dict):
        res = dict()
        for k, v in config.items():
            if k in sp_config:
                res[k] = dict_chain(sp_config[k], v, __deep= __deep+1)
            else:
                res[k] = v
    elif isinstance(config, list):
        res = []
        for index, i in enumerate(config):
            res[index] = dict_chain(sp_config[index], i, __deep= __deep+1)
    else:
        res = sp_config
    return  res


def start_tuning(config):
    storage_path = join(*config['saving_data_params']['tune_results_path'])
    exp_name = ''.join(config['saving_data_params']['tune_results_dir'])
    # exp_name = "tune_analyzing_results_20240410-090935"
    # search_space = {
    #                    'lr': 1e-3,
    #                    'batch_size': 16,
    #                    'num_workers': 1,
    #                    'alpha': 0 * 0.25,
    #                    "gamma": 0 * 2.0,
    #                    "reduction": "mean",
    #                    "from_logits": False,
    #                    "padding_sec": PADDING_SEC,
    #                    "is_tune": False,
    #                    "enable_tune_features": False,
    #                    "conv_lr": 1e-3,
    #                    'layer_1_size': 2048,
    #                    'layer_2_size': 1024,
    #                    'patch_transformer_size': 16,
    #                    'transformer_depth': 6,
    #                    'transformer_attantion_head_count': 16
    #                } | {
    #                    "layer_1_size": tune.choice([32, 64, 128, 256, 512, ]),
    #                    "layer_2_size": tune.choice([32, 64, 128, 256, 512, ]),
    #                    "lr": tune.loguniform(1e-4, 1e-1),
    #                    "batch_size": tune.choice([8, 16, 32]),
    #                    # "gamma": tune.choice([i / 100 * 2 + 0.1 for i in range(100)]),
    #                    # "conv_h_count": tune.choice([0, 1, 2, 4, 8, 16]),
    #                    # "conv_w_count": tune.choice([0, 1, 2, 4, 8, 16]),
    #                    "num_workers": 15,
    #                    "is_tune": True,
    #                    "enable_tune_features": False,
    #                    "patch_transformer_size": tune.choice([2 ** i for i in range(2, 7)]),
    #                    "transformer_depth": tune.choice(list(range(2, 10))),
    #                    'transformer_attantion_head_count': tune.choice(list(range(2, 24, 2))),
    #                }

    search_space = {
        "model_architecture": {
            "conv_h_count":  tune.choice([0, 1, 2, 4, 8, 16]),
            "conv_w_count":  tune.choice([0, 1, 2, 4, 8, 16]),
            "layer_1_size": tune.choice([32, 64, 128, 256, 512, ]),
            "layer_2_size": tune.choice([32, 64, 128, 256, 512, ]),
            "patch_transformer_size": tune.choice([2 ** i for i in range(2, 7)]),
            "transformer_depth": tune.choice(list(range(2, 10))),
            'transformer_attantion_head_count': tune.choice(list(range(2, 24, 2))),
        },
        "learn_params": {
            "lr": {
                "base_lr": tune.loguniform(1e-4, 1e-1),
                "conv_lr": tune.loguniform(1e-4, 1e-1),
            },
            "batch_size": tune.choice([8, 16, 32]),
        }
    }

    search_space = dict_chain(search_space, config)

    # The maximum training epochs
    num_epochs = config['tune']["max_tune_epochs"]
    # Number of sampls from parameter space
    num_samples = config['tune']['num_samples']
    # scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=config['load_dataset_workers_num'],
        use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        name=exp_name,
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val/acc",
            checkpoint_score_order="max",
        ),
        storage_path=storage_path,
        callbacks=[DeleteCallback()],
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    # checkpoint = Checkpoint("s3://bucket/ckpt_dir")

    ray_trainer = TorchTrainer(
        tune_wapper_of_train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # results = tune_asha(
    #     ray_trainer,
    #     search_space,
    #     num_samples=num_samples,
    #     num_epochs=num_epochs
    # )
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    ray.init(_temp_dir=join(config['saving_data_params']['tune_session_path']))
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val/acc",
            mode="max",
            num_samples=max_epoch,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    print(results)
