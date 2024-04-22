
import torch
import numpy as np
import torch.nn.functional as F


class SigmoidFocalCrossEntropy(torch.nn.Module):
    def __init__(self,
                 from_logits: bool = False,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'none',
                 device='cpu',
                 dataset=None,
                 ):
        super(SigmoidFocalCrossEntropy, self).__init__()
        self.device = device
        assert dataset and bool(dataset)
        if gamma and gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")

        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        class_counts = np.bincount([i['emotion'] for i in dataset])
        num_classes = len(class_counts)
        total_samples = len(dataset)

        self.class_weights = []
        for count in class_counts:
            weight = 1 / (count / total_samples)
            self.class_weights.append(weight)
        # self.alpha =  torch.FloatTensor(self.class_weights).to('cuda:0')

    def forward(self, inputs, targets):
        # print("targets", targets)
        # print("inputs", inputs)
        # print("self.alpha", self.alpha)
        # print("inputs!!!!!!!!", inputs.size(), targets.size(),)
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)

        if self.from_logits:
            inputs = torch.softmax(inputs, dim=1)

        inputs = torch.amax(inputs, 1)
        # print("inputs!!!!!!!!", inputs.size(), targets.size(), self.alpha, self.gamma)
        p_t = (targets * inputs) + ((1 - targets) * (1 - inputs))
        alpha_factor = 1.0
        modulating_factor = 1.0

        if True or  self.alpha:
            alpha_factor = (targets * self.alpha
                            + (1 - targets)
                            * (1 - self.alpha))

        if True or  self.gamma:
            modulating_factor = torch.pow((1.0 - p_t), self.gamma)

        # pt = torch.exp(-ce_loss)
        # loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()

        res_loss = torch.mean(alpha_factor * modulating_factor * ce_loss, axis=-1)
        return res_loss
