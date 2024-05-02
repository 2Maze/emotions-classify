from os.path import join, dirname
from collections import OrderedDict

import torchvision.models as models
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers import Wav2Vec2Model, Wav2Vec2Config

from config.constants import ROOT_DIR, PADDING_SEC


class SpectrogramCnnClassifier(nn.Module):

    def __init__(
            self,
            num_classes,
            # padding_sec,
            config: dict | None = None,
            dataset=None,
            **k_,
    ):
        super(SpectrogramCnnClassifier, self).__init__()
        self.padding_sec = config['learn_params']['padding_sec']
        self.padding_sec_w = 50 * self.padding_sec - 1

        self.config = config
        self.spectrogram_size = config['learn_params']['spectrogram_size']

        self.efficientnet_model = models.efficientnet_b0(weights=False)

        self.efficientnet_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                           bias=False)
        self.efficientnet_model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

        # for param in self.embedding_model.wav2vec2.parameters():
        #     param.requires_grad = False

        self.classifier = self.efficientnet_model

    def forward(
            self,
            X: torch.Tensor,
            mask_time_indices: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
    ):
        # features = self.get_embeddings(X, mask_time_indices=mask_time_indices, attention_mask=attention_mask)
        logits = self.classifier(X.view(-1, 1, self.spectrogram_size, self.spectrogram_size))
        logits = torch.softmax(logits, dim=1)
        return logits

    def get_embeddings(
            self,
            input_values: torch.Tensor,
            mask_time_indices: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = self.embedding_model.wav2vec2(input_values).extract_features
        return hidden_states
