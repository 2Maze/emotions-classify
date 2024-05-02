# emotions-classify

# Use

## Install

1. Clone this repo

2. Unzip `data.tar.gz` file to `<root_project>/data` folder.

3. Create config.json file (as example you can be use `<project_root>/configs/example.config.json`)

## Build docker image

1. Copy [entrypoint.d](https://gitlab.com/nvidia/container-images/cuda/-/tree/master/entrypoint.d) to `<project_root>/docker/source`
2. Copy [NGC-DL-CONTAINER-LICENSE](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/NGC-DL-CONTAINER-LICENSE) to `<project_root>/docker/source`
3. Copy [nvidia_entrypoint.sh](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/nvidia_entrypoint.sh) to `<project_root>/docker/source`
4. Run from `project_root` directory
```bash
docker build -f  ./docker/ighting.Dockerfile --build-arg REQUIREMENTS_FILE=cu_12_2.txt . -t daniinxorchenabo/emotions-classify-env:lighting-cu122-latest
```

## Run

### Run training with config, tensorboard server and jupiter server
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 0.0.0.0:8888:8888 -p 0.0.0.0:6006:6006 -it -v .:/workspace/NN  daniinxorchenabo/emotions-classify-env:lighting-cu122-latest ./docker/before_learn.sh train.py train --config-file ./configs/test.config.json 
```

### Run training with config
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it -v .:/workspace/NN  daniinxorchenabo/emotions-classify-env:lighting-cu122-latest python train.py train --config-file ./configs/config.json 
```

## Config format


Пример конфига для обучения модели


```json
{
  "pipeline": [
    {
      "type": "train",
      "model": "Wav2Vec2FnClassifier",
      "model_architecture": {
        "conv_h_count": 1,
        "conv_w_count": 1,
        "layer_1_size": 32,
        "layer_2_size": 32
      },
      "learn_params": {
        "lr": {
          "base_lr": 0.0007505403591860749,
          "conv_lr": 0.0007505403591860749
        },
        "batch_size": 8,
        "padding_sec": 5,
        "max_epoch": 10
      },
      "saving_data_params": {
        "saved_checkpoints_path": [
          "$$ROOT_DIR$$",
          "weights",
          "checkpoints",
          "$$MODEL_NAME$$",
          "$$DATETIME$$"
        ],
        "saved_checkpoints_filename": [
          "classifier_",
          "$$DATETIME$$",
          "_{epoch:02d}"
        ],
        "tensorboard_lr_monitors_logs_path": [
          "$$ROOT_DIR$$",
          "logs"
        ],
        "start_from_saved_checkpoint_path": null
      }
    }
  ],
  "load_dataset_workers_num": 1
}
```

Пример конфига для тюнинга моделей


```json
{
  "pipeline": [
    {
      "type": "tune",
      "model": "TransformerClassifier",
      "model_architecture": {
        "layer_1_size": null,
        "layer_2_size": null,
        "patch_transformer_size": null,
        "transformer_depth": null,
        "transformer_attantion_head_count": null
      },
      "learn_params": {
        "lr": {
          "base_lr": null
        },
        "batch_size": null,
        "spectrogram_size": 512,
        "max_epoch": 200
      },
      "saving_data_params": {
        "saved_checkpoints_path": [
          "$$ROOT_DIR$$",
          "weights",
          "checkpoints",
          "$$DATETIME$$"
        ],
        "saved_checkpoints_filename": [
          "classifier_",
          "$$DATETIME$$",
          "_{epoch:02d}"
        ],
        "tensorboard_lr_monitors_logs_path": [
          "$$ROOT_DIR$$",
          "logs"
        ],
        "tune_results_path": [
          "$$ROOT_DIR$$",
          "tmp",
          "tune"
        ],
        "tune_results_dir": [
          "tune_",
          "$$DATETIME$$"
        ],
        "tune_session_path": [
          "$$ROOT_DIR$$",
          "tmp",
          "ray",
          "session"
        ]
      },
      "tune": {
        "enable_tune_features": false,
        "max_tune_epochs": 20,
        "num_samples": 100
      }
    }
  ],
  "load_dataset_workers_num": 1
}
```
_Установка значения `null` помечает параметр для тюнинга. В противном случае (в случае установки конкретного значения) тюнинг по этому параметру произодиться не будет_


Пример конфига для вывода результатов тюнинга.


```json
{
  "pipeline": [
    {
      "type": "print_tune_res",
      "res_path": [
        "$$ROOT_DIR$$",
        "tmp",
        "tune",
        "tune_20240425-101338"
      ]
    }
  ],
  "load_dataset_workers_num": 1
}
```

* `pipeline`.`type` - тип действия. Возможные значения:
  * `train` - Запустить обучение модели с заданной конфигурацией
  * `tune` - Запустить тюнинг для заданной модели
  * `print_tune_res` - Напечатать результаты тюнинга


* `pipeline`.`model` - Тип нейросетевой архитектуры. Используется только для `pipeline`.`type`=`train` и `pipeline`.`type`=`tune`. Принимает следующие значения:
  * `Wav2Vec2FnClassifier` - Использование полносвязной архитектуры, с фича-экстрактором на базе `Wav2Vec`
  * `Wav2Vec2CnnClassifier` - Использование модели `efficientnet` для обработки ембедингов `Wav2Vec`
  * `SpectrogramCnnClassifier` - Использование модели `efficientnet` для обработки спектрограмм
  * `TransformerClassifier` -  Использование модели `VisionTransformer` для обработки спектрограмм


* `pipeline`.`learn_params` - Набор параметров, использующихся для обучения. Применяется в `pipeline`.`type`=`train` и `pipeline`.`type`=`tune`
* `pipeline`.`learn_params`.`lr`.`base_lr` - Основной `learning rate` для обучения модели.
* `pipeline`.`learn_params`.`lr`.`conv_lr` - Отдельный `learning rate` для обучения `conv` слоёв. Применяется для обучения `pipeline`.`model`=`Wav2Vec2FnClassifier`
* `pipeline`.`learn_params`.`batch_size` - Размер батча для обучения нейронной сети
* `pipeline`.`learn_params`.`max_epoch` - Максимальное количество эпох для обучения. Имеет смысл только при `pipeline`.`type`=`train`
* `pipeline`.`learn_params`.`padding_sec` - Количество секунд, по которому обрезаются данные из обучающей выборки. Используется только в нейросетевых моделях, основанных на `Wav2Vec` подходе, таких как `pipeline`.`model`=`Wav2Vec2FnClassifier` и `pipeline`.`model`=`Wav2Vec2CnnClassifier` 
* `pipeline`.`learn_params`.`spectrogram_size` - Размер спектограммы в пикселях. Используется только в нейросетевых моделях, основанных на спектограммах, таких как `pipeline`.`model`=`SpectrogramCnnClassifier` и `pipeline`.`model`=`TransformerClassifier`


* `pipeline`.`saving_data_params` - Пути для сохранения конфигов, логов и так далее. Используется только в `pipeline`.`type`=`train` и `pipeline`.`type`=`tune`.
* `pipeline`.`saving_data_params`.`saved_checkpoints_path` - Путь для сохранения весов нейросетей. Имеет значение только для `pipeline`.`type`=`train`. Значение является массивом из названий папок.
* `pipeline`.`saving_data_params`.`saved_checkpoints_filename` - Шаблон имени файла весов нейросетей. Имеет значение только для `pipeline`.`type`=`train`. Значение явялется массивом, который конкатенируется в строку.
* `pipeline`.`saving_data_params`.`tensorboard_lr_monitors_logs_path` - Путь для сохранения логов обучения? rjnjhst визуализируются при помощи `tensorboard`. Значение является массивом из названий папок.
* `pipeline`.`saving_data_params`.`start_from_saved_checkpoint_path` - Путь, по которому расположены сохранённые веса, для того, чтобы продолжить обучение именно с них.
  * Если установленно `null`, то обучение будет производиться с начсала.
  * Если значение - массив из названий папок, то обучение будет происходить с чекпоинта, указанного в этом массиве.
* `pipeline`.`saving_data_params`.`tune_results_path` - Путь, по которому будут сохраняться результаты тюнинга. Имеет смысл только при `pipeline`.`type`=`tune`. Значение является массивом из названий папок.
* `pipeline`.`saving_data_params`.`tune_results_dir` - Формат названия папки, в которой будут храниться результаты тюнинга. Имеет смысл только при `pipeline`.`type`=`tune`. Значение явялется массивом, который конкатенируется в строку.
* `pipeline`.`saving_data_params`.`tune_session_path` - Путь, по которому будут сохраняться файлы сессий тюнинга. Имеет смысл только при `pipeline`.`type`=`tune`. Значение является массивом из названий папок.


* `pipeline`.`tune` - Специфичные параметры для тюнинга.
* `pipeline`.`tune`.`enable_tune_features` - Включить оптимизации `ray` для тюнинга моделей
  * `true` - включить
  * `false` - выключить
* `pipeline`.`tune`.`max_tune_epochs` - Максимальное количество эпох для тюнинга одной конфигурации
* `pipeline`.`tune`.`num_samples` - Количество моделей, среди которых будет производиться тюнинг


* `load_dataset_workers_num` - Количество процессов для загрузки данных


## model_architecture params

Параметры, отвечающие за архитектуру нейронной сети.

* `pipeline`.`model`=`Wav2Vec2FnClassifier`
  * `pipeline`.`model_architecture`.`conv_h_count` - Количество `1d` свёрток по высоте эмбелингов `Wav2Vec`
  * `pipeline`.`model_architecture`.`conv_w_count` - Количество `1d` свёрток по линии времени эмбелингов `Wav2Vec`
  * `pipeline`.`model_architecture`.`layer_1_size` - Количество нейронов на первом слое классификатора
  * `pipeline`.`model_architecture`.`layer_2_size` - Количество нейронов на втором слое классификатора
* `pipeline`.`model`=`Wav2Vec2CnnClassifier`
  * Не имеет настраиваемых параметров
* `pipeline`.`model`=`SpectrogramCnnClassifier`
  * Не имеет настраиваемых параметров
* `pipeline`.`model`=`TransformerClassifier`
  * `pipeline`.`model_architecture`.`layer_1_size` - Последнее измерение выходного тензора после линейного преобразования
  * `pipeline`.`model_architecture`.`layer_2_size` - Размер слоя MLP (FeedForward)
  * `pipeline`.`model_architecture`.`patch_transformer_size` - Размер патчей (patch_size). значение image_size (hfpvth входной спектограммы) должно быть кратно значению patch_size. Количество патчей: n = (image_size // patch_size) ** 2 и n должно быть больше 16.
  * `pipeline`.`model_architecture`.`transformer_depth` - Количество блоков трансформера.
  * `pipeline`.`model_architecture`.`transformer_attantion_head_count` - Количество параллельных слоёв внимания в одном блоке трансформера.


