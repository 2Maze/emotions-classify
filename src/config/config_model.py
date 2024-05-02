from __future__ import annotations

from typing import TypedDict, TypeVar
from typing_extensions import Unpack


class AllConfig(TypedDict):
    pipeline: str


NN_MODEL = TypeVar(
    Literal['Wav2Vec2FnClassifier']
    | Literal['Wav2Vec2CnnClassifier']
    | Literal['SpectrogramCnnClassifier']
    | Literal['TransformerClassifier']
)

class BaseUnarConfig(TypedDict):
    type: Literal['train'] | Literal['tune'] |  Literal['print_tune_res'],

    model_architecture: None
    learn_params: None
    saving_data_params: None

class BaseTrainConfig(BaseUnarConfig):
    type: Literal['train']
    model: NN_MODEL

class BaseTuneConfig(BaseUnarConfig):
    type: Literal['tune']
    model: NN_MODEL

class BasePrintResConfig(BaseUnarConfig):
    pass

