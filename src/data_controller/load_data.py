from os.path import join

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torchaudio.transforms import Spectrogram

from config.constants import ROOT_DIR, PADDING_SEC
from data_controller.emotion_dataset import EmotionDataset


def collate_fn(items):
    output = {key: [] for key in list(items[0].keys())}
    for item in items:
        for key in item:
            if isinstance(item[key], torch.Tensor):
                output[key].append(item[key])
            else:
                output[key].append(torch.tensor(item[key]))
    for key in list(output.keys()):
        if key == 'emotion' or key == 'state':
            output[key] = torch.stack(output[key])
        else:
            # print(key, list(output), output[key])
            output[key] = pad_sequence(output[key], batch_first=True)
    return output


def load_data(
        bath_size=10,
        validation_bath_size=None,
        num_workers=1,
        test_size=0.15,
        dataset_class=EmotionDataset,
        spectrogram_size=512,
        padding_sec=PADDING_SEC,
):
    validation_bath_size = validation_bath_size or bath_size

    dataset = dataset_class(
        annotation=join(ROOT_DIR, 'data', 'dataset', 'annotations.json'),
        padding_sec=padding_sec,
        spectrogram_size=spectrogram_size,
    )
    dataset = dataset

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=dataset.emotion_labels
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bath_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=validation_bath_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True
    )

    return train_dataloader, val_dataloader, dataset
