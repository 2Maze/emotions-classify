import torch.nn as nn
import torchaudio


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Wav2Vec2Classifier, self).__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.feature_extractor = bundle.get_model()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(768, 1024, bias=True),
        				nn.ReLU(inplace=True),
        				nn.Linear(1024, 512, bias=True),
        				nn.ReLU(inplace=True),
        				nn.Linear(512, num_classes, bias=True))

    def forward(self, X):
        features = self.get_embeddings(X)
        logits = self.classifier(features)
        return logits

    def get_embeddings(self, X):
        embeddings = self.feature_extractor(X)[0].mean(axis=1)
        return nn.functional.normalize(embeddings)
