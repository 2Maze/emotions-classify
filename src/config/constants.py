from os.path import split, dirname, join
from functools import reduce

_FILE_NESTED_LEVEL = 2
ROOT_DIR = reduce(lambda i, _: split(i)[0], range(_FILE_NESTED_LEVEL), dirname(__file__))
print(ROOT_DIR)
