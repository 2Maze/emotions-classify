import librosa
import json
import os

from torch.utils.data import Dataset


EMOTIONS = ['гнев', 'нейтраль', 'отвращение', 'печаль', 'радость', 'страх', 'удивление']
STATES = ['нейтральное', 'отрицательное', 'положительное']


class EmotionDataset(Dataset):
    def __init__(self,
                 annotation: str):
        assert os.path.exists(annotation), 'Annotation file does not exist!'
        with open(annotation) as file:
            self.annotation = json.load(file)

        self.emotions = EMOTIONS
        self.states = STATES
        self.emotion_labels = [self.emotions.index(self.annotation[key]['emotion']) for key in self.annotation]
        self.state_labels = [self.states.index(self.annotation[key]['state']) for key in self.annotation]

        self.folder = os.path.split(annotation)[0]
        self.annotation_files = list(self.annotation.keys())

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        key = self.annotation_files[idx]
        speech_array, sampling_rate = librosa.load(os.path.join(self.folder, key), sr=16_000)
        return {'array': speech_array,
                'emotion': self.emotions.index(self.annotation[key]['emotion']),
                'state': self.states.index(self.annotation[key]['state'])}
