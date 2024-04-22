from datetime import datetime
from os.path import join

import torch
from pytorch_lightning.callbacks import Checkpoint
from ray import tune
from ray.air import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from src.config.constants import ROOT_DIR
from src.train.callbacks.delete_checkpoint import DeleteCallback
from src.train.train_func import train_func


def tune_wapper_of_train_func(config):
    torch.cuda.empty_cache()
    tune.utils.wait_for_gpu()
    try:
        result = train_func(
            config,
            tuning=True,
        )
    except Exception as e:
        print("some error win train")
        torch.cuda.empty_cache()
        tune.utils.wait_for_gpu()
        raise e from e


def start_tuning():
    storage_path = join(ROOT_DIR, 'tmp', 'tune')
    exp_name = "tune_analyzing_results_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # exp_name = "tune_analyzing_results_20240410-090935"
    search_space = {
        # "layer_1_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        # "layer_2_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([4, 8, 16, 32, 64]),
        "gamma": tune.choice([i / 100 * 2 + 0.1 for i in range(100)]),
        # "conv_h_count": tune.choice([0, 1, 2, 4, 8, 16]),
        # "conv_w_count": tune.choice([0, 1, 2, 4, 8, 16]),
        "num_workers": 1
    }
    # The maximum training epochs
    num_epochs = 20
    # Number of sampls from parameter space
    num_samples = 100
    # scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=1,
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

    # ray.init(_temp_dir=join(ROOT_DIR, 'tmp', "ray", "session"))
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val/acc",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    print(results)
