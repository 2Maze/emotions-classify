import librosa
import json
import os
import numpy as np

from torch.utils.data import Dataset
from typing import Optional


EMOTIONS = ['гнев', 'нейтраль', 'отвращение', 'печаль', 'радость', 'страх', 'удивление']
STATES = ['нейтральное', 'отрицательное', 'положительное']
SAMPLING_RATE = 16_000


class EmotionDataset(Dataset):
    def __init__(self,
                 annotation: str,
                 padding_sec: Optional[int] = None):
        assert os.path.exists(annotation), 'Annotation file does not exist!'
        with open(annotation) as file:
            self.annotation = json.load(file)

        self.emotions = EMOTIONS
        self.states = STATES
        self.emotion_labels = [self.emotions.index(self.annotation[key]['emotion']) for key in self.annotation]
        self.state_labels = [self.states.index(self.annotation[key]['state']) for key in self.annotation]

        self.folder = os.path.split(annotation)[0]
        self.annotation_files = list(self.annotation.keys())
        self.padding_sec = padding_sec
        if padding_sec:
            self.target_length_samples = padding_sec * SAMPLING_RATE
        else:
            self.target_length_samples = None

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        key = self.annotation_files[idx]
        speech_array, sampling_rate = librosa.load(os.path.join(self.folder, key), sr=16_000)
        assert sampling_rate == SAMPLING_RATE, 'Sampling rate input audio, is not correct!'
        if self.padding_sec is not None:
            if len(speech_array) < self.target_length_samples:
                speech_array = np.concatenate((speech_array,
                                              np.zeros(int(self.target_length_samples - len(speech_array)))),
                                              axis=0,
                                              dtype=np.float32)
            else:
                speech_array = speech_array[:self.target_length_samples]

        return {'array': speech_array,
                'emotion': self.emotions.index(self.annotation[key]['emotion']),
                'state': self.states.index(self.annotation[key]['state'])}