import os
from os.path import split, dirname, join
from functools import reduce

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "0"

_FILE_NESTED_LEVEL = 2
ROOT_DIR = reduce(lambda i, _: split(i)[0], range(_FILE_NESTED_LEVEL), dirname(__file__))
print(ROOT_DIR)
PADDING_SEC = 5
