import json
import os
import shutil
import traceback
from os.path import split, join

from ray.tune.callback import Callback
from ray.tune.experiment import Trial


class DeleteCallback(Callback):

    @staticmethod
    def get_train_num_from_dir(path):
        return int(split(join("some", path))[1].split('_')[3].strip())

    def on_trial_complete(
            self, iteration: int, trials: list[Trial], trial: Trial, **info,
    ):
        last_result = trial.last_result
        # Filter out which checkpoints to delete on trial completion
        print("___deleting checkpoint", last_result, trial, trial.path, trial.local_path)
        if True and last_result["val/acc"] < 0.75:
            try:
                tune_root_dir, curr_dir = split(trial.path)
                curr_tune_num = self.get_train_num_from_dir(trial.path)
                all_delete_dirs = [
                    (i, directory, checkpoint_dir)
                    for (i, directory, checkpoint_dir) in (
                        (i, directory, j)
                        for (i, directory) in (
                        (self.get_train_num_from_dir(join(tune_root_dir, f)), join(tune_root_dir, f))
                        for f in os.listdir(tune_root_dir)
                        if os.path.isdir(join(tune_root_dir, f))
                    ) if 0 < curr_tune_num - i < 10
                        for j in os.listdir(directory)
                        if j.startswith('checkpoint_')
                    )]

                print(all_delete_dirs)
                checkpoint_path = join(trial.path, last_result["checkpoint_dir_name"])
                for (_, one_tune_iteration_dir, checkpoint_dir) in all_delete_dirs:
                    results_dir = join(one_tune_iteration_dir, "result.json")
                    with open(results_dir, "r") as f:
                        results = [i['checkpoint_dir_name'] for i in (json.loads(i) for i in f.readlines()) if
                                   i['val/acc'] > 0.75]
                    print("one_tune_iteration_dir", one_tune_iteration_dir, checkpoint_dir, split(checkpoint_dir)[1],
                          last_result["val/acc"])
                    if split(checkpoint_dir)[1] not in results:
                        print("deleting", join(one_tune_iteration_dir, checkpoint_dir))
                        shutil.rmtree(join(one_tune_iteration_dir, checkpoint_dir))
            except Exception as exp:
                print(f"Unable to delete checkpoint of {trial}", exp, "\n", traceback.format_exc())
