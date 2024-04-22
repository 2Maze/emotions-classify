import torch
import numpy as np
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, dataset=None, device='cpu'):
        super(FocalLoss, self).__init__()
        self.device = device
        assert bool(dataset)

        self.gamma = gamma

        class_counts = np.bincount([i['emotion'] for i in dataset])
        num_classes = len(class_counts)
        total_samples = len(dataset)

        self.class_weights = []
        for count in class_counts:
            weight = 1 / (count / total_samples)
            self.class_weights.append(weight)
        self.alpha = torch.FloatTensor(self.class_weights).to('cuda:0')

    def forward(self, inputs, targets):
        # print("targets", targets)
        # print("inputs", inputs)
        # print("self.alpha", self.alpha)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss, ce_loss
