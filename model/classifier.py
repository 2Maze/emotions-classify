import torch.nn as nn
import torchaudio


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Wav2Vec2Classifier, self).__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.feature_extractor = bundle.get_model()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(768, num_classes)

    def forward(self, X):
        features = self.get_embeddings(X)
        logits = self.linear(features)
        return logits

    def get_embeddings(self, X):
        embeddings = self.feature_extractor(X)[0].mean(axis=1)
        return nn.functional.normalize(embeddings)
