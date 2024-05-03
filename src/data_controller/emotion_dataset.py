import librosa
import json
import os
import numpy as np
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset

EMOTIONS = ['гнев', 'нейтраль', 'отвращение', 'печаль', 'радость', 'страх', 'удивление']
EMOTIONS.remove('отвращение')
STATES = ['нейтральное', 'отрицательное', 'положительное']
SAMPLING_RATE = 16_000


class EmotionDataset(Dataset):
    def __init__(
            self,
            annotation: str,
            padding_sec: Optional[int] = None,
            spectrogram_size: Optional[int] = None,
    ):
        assert os.path.exists(annotation), 'Annotation file does not exist!'
        with open(annotation) as file:
            self.annotation = json.load(file)
        self.annotation = {k: v for k, v in self.annotation.items()
                           if k.removesuffix(".wav").split("_")[-1] != "0"
                           and v['emotion'] != 'отвращение'
                           }
        self.emotions = EMOTIONS
        self.states = STATES
        self.emotion_labels = [self.emotions.index(self.annotation[key]['emotion']) for key in self.annotation]
        self.state_labels = [self.states.index(self.annotation[key]['state']) for key in self.annotation]

        self.folder = os.path.split(annotation)[0]
        self.annotation_files = list(self.annotation.keys())
        self.padding_sec = padding_sec
        if padding_sec:
            self.spectrogram_size = padding_sec * SAMPLING_RATE
        else:
            self.spectrogram_size = None

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        key = self.annotation_files[idx]
        speech_array, sampling_rate = librosa.load(os.path.join(self.folder, key), sr=16_000)
        assert sampling_rate == SAMPLING_RATE, 'Sampling rate input audio, is not correct!'
        if self.padding_sec is not None:
            if len(speech_array) < self.spectrogram_size:
                speech_array = np.concatenate(
                    (speech_array,
                     np.zeros(int(self.spectrogram_size - len(speech_array)))),
                    axis=0,
                    dtype=np.float32
                )
            else:
                speech_array = speech_array[:self.spectrogram_size]

        return {'array': speech_array,
                'emotion': self.emotions.index(self.annotation[key]['emotion']),
                'state': self.states.index(self.annotation[key]['state'])}


class EmotionSpectrogramDataset(EmotionDataset):

    def __init__(
            self,
            *args,
            spectrogram_size=512,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spectrogram_size = spectrogram_size
        self.fit = ( spectrogram_size * 2 - 2)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=self.fit)

    def __getitem__(self, idx):
        key = self.annotation_files[idx]

        waveform, sample_rate = torchaudio.load(os.path.join(self.folder, key), normalize=True)
        spectrogram = self.spectrogram_transform(waveform)
        # print("^^1", spectrogram.size())
        spectrogram = spectrogram.view(self.spectrogram_size, -1)
        # print("^^2", spectrogram.size())
        if self.padding_sec is not None:
            if spectrogram.size()[1] < self.spectrogram_size:
                speech_array = torch.cat(
                    (spectrogram,
                     torch.zeros(spectrogram.size()[0], int(self.spectrogram_size - spectrogram.size()[1]))),
                    1
                    # dtype=np.float32
                )
            elif spectrogram.size()[1] > self.spectrogram_size:
                speech_array = spectrogram[:, :self.spectrogram_size]
            else:
                speech_array = spectrogram
        # print("^^3", speech_array.size())
        return {'array': speech_array,
                'emotion': self.emotions.index(self.annotation[key]['emotion']),
                'state': self.states.index(self.annotation[key]['state'])}
