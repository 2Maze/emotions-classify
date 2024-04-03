import torch
import torchmetrics
import numpy as np
import lightning as L
import torch.nn.functional as F

from src.utils.data import EmotionDataset
from src.model import Wav2Vec2Classifier
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor


class LitModule(L.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = Wav2Vec2Classifier(num_classes)

        self.val_loss = []
        self.acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['array'])
        loss = F.cross_entropy(outputs, batch['emotion'])
        self.log('train/loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['array'])
        loss = F.cross_entropy(outputs, batch['emotion'])
        self.val_loss.append(loss.item())
        self.log('val/acc', self.acc(outputs, batch['emotion']), on_step=False, on_epoch=True)
        self.log('val/f1', self.f1(outputs, batch['emotion']), on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self):
        self.log('val/mean_loss', np.array(self.val_loss).mean())
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
        return [optimizer], [scheduler]


def collate_fn(items):
    output = {key: [] for key in list(items[0].keys())}
    for item in items:
        for key in item:
            output[key].append(torch.tensor(item[key]))
    for key in list(output.keys()):
        if key == 'emotion' or key == 'state':
            output[key] = torch.stack(output[key])
        else:
            output[key] = pad_sequence(output[key], batch_first=True)
    return output


def main():
    dataset = EmotionDataset(annotation='data/dataset/annotations.json')

    train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                          test_size=0.15,
                                          random_state=42,
                                          shuffle=True,
                                          stratify=dataset.emotion_labels)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename="classifier_{epoch:02d}",
        every_n_epochs=10,
        save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = LitModule(len(dataset.emotions))
    trainer = L.Trainer(accelerator='gpu',
                        devices=1,
                        max_epochs=100,
                        callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
