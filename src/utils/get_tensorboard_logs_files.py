import os
from os.path import join, split, dirname
from functools import reduce
# from config.constants import ROOT_DIR

_FILE_NESTED_LEVEL = 2
ROOT_DIR = reduce(lambda i, _: split(i)[0], range(_FILE_NESTED_LEVEL), dirname(__file__))

if __name__ == "__main__":
    train_folders = [
        f"{f}:{join(ROOT_DIR, 'logs', f)}"
        for f in os.listdir(join(ROOT_DIR, "logs"))
        if os.path.isdir(join(ROOT_DIR, "logs"))
    ]
    tune_folders = [
        f"{cl_name}_{full_path3}:{join(full_path, full_path2, full_path3, 'driver_artifacts')}"
        for full_path, cl_name in
        [
            [join(ROOT_DIR, "tmp", 'ray', f, f2, 'artifacts'), f]
            for f in os.listdir(join(ROOT_DIR, "tmp", 'ray' ))
            if os.path.isdir(join(ROOT_DIR, "tmp", 'ray',  f))
            for f2 in os.listdir(join(ROOT_DIR, "tmp", 'ray',  f))
            if os.path.isdir(join(ROOT_DIR, "tmp", 'ray',  f, f2, 'artifacts'))
        ]
        for full_path2 in os.listdir(full_path)
        if os.path.isdir(join(full_path, full_path2))
        for full_path3 in os.listdir(join(full_path, full_path2))
        if os.path.isdir(join(full_path, full_path2, full_path3, "driver_artifacts"))
    ]
    print(*tune_folders, *train_folders, sep=',')

