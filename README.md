# emotions-classify

# Dataset load

1. Unzip `data.tar.gz` file to `<root_project>/data` folder.

# Build

## as docker

```bash
docker build --build-arg REQUIREMENTS_FILE=cu_12_2.txt . -t daniinxorchenabo/emotions-classify:cu_12_2
```

# Run jupiter server

## as docker

1. Run docker image
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 8888:8888 -it -v .:/workspace/NN  daniinxorchenabo/emotions-classify:cu_12_2 jupyter lab  --allow-root  --ip=0.0.0.0 
```

# Run python scripts

## as docker

1. Run docker image
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 8888:8888 -it -v .:/workspace/NN  daniinxorchenabo/emotions-classify:cu_12_2 python <some_file>.py
```
