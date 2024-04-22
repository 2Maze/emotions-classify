import lightning as L
import numpy as np
import torch.nn.functional as F
import torchmetrics
import torch

from src.metrics.confusion_matrix import CreateConfMatrix
from src.model import Wav2Vec2CnnClassifier


class LitModule(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            learning_rate: float = 1e-3,
            conv_learning_rate: float = 1e-3,
            conv_h_count: int = 0,
            conv_w_count: int = 0,
            padding_sec: int = 0,
            layer_1_size: int = 1024,
            layer_2_size: int = 512,
            h_mean_enable: bool = True,
            w_mean_enable: bool = True,
            config: dict | None = None,
            lables: list[str] = None,
            is_tune: bool = False,
            dataset: list = None,
            gamma: float = 2.0
    ):
        super().__init__()
        self.model = Wav2Vec2CnnClassifier(
            num_classes,
            padding_sec,
            conv_h_count=conv_h_count,
            conv_w_count=conv_w_count,
            layer_1_size=layer_1_size,
            layer_2_size=layer_2_size,
            config=config
        )
        self.config = config
        self.is_tune = is_tune
        self.num_classes = num_classes

        self.lr = learning_rate
        self.conv_lr = conv_learning_rate

        self.val_loss = []
        self.val_pred = []
        self.val_y_true = []
        self.train_y_pred = []
        self.train_y_true = []
        self.acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=num_classes)
        self.confusion_matrix = CreateConfMatrix(num_classes, lables, self.logger)

        self.class_weights = []
        class_counts = np.bincount([i['emotion'] for i in dataset])
        num_classes = len(class_counts)
        total_samples = len(dataset)
        for count in class_counts:
            weight = 1 / (count / total_samples)
            self.class_weights.append(weight)
        self.focal_loss_w = torch.FloatTensor(self.class_weights).to('cuda:0')

        # self.focal_loss = FocalLoss(dataset=dataset, device=self.device, gamma=gamma).to(self.device)
        # self.focal_loss = tfa.losses.SigmoidFocalCrossEntropy()
        # self.focal_loss = SoftmaxFocalLoss(
        #     # from_logits=config["from_logits"],
        #     gamma=config["gamma"],
        #     alpha=config["alpha"],
        #     reduction=config["reduction"],
        #     weight= torch.FloatTensor(self.class_weights).to('cuda:0'),
        #     # device=self.device,
        #     # dataset=dataset,
        # )
        self.focal_loss = None


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['array'])
        ce_loss = F.cross_entropy(outputs, batch['emotion'])
        # tfa.losses.SigmoidFocalCrossEntropy()
        if self.focal_loss is not None:
            loss = self.focal_loss(outputs, batch['emotion'])
        else :
            loss = ce_loss
        self.log('train/ce_loss', ce_loss, on_step=True, on_epoch=False)
        self.log('train/focal_loss', loss, on_step=True, on_epoch=False)
        if self.is_tune is False:
            self.train_y_pred.append(outputs)
            self.train_y_true.append(batch['emotion'])

        return loss

    def on_train_epoch_end(self):
        if self.is_tune is False:
            preds = torch.cat([i for i in self.train_y_pred])
            targets = torch.cat([i for i in self.train_y_true])

            self.confusion_matrix.draw_confusion_matrix(preds, targets, self.current_epoch, self.logger,
                                                        "train/Confusion matrix")

            self.train_y_pred = []
            self.train_y_true = []

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['array'])
        # print("---outputs", outputs.size(), outputs.max(dim=1), batch['emotion'])
        ce_loss = F.cross_entropy(outputs, batch['emotion'])
        # print("ce_loss", ce_loss)
        if self.focal_loss is not None:
            loss = self.focal_loss(outputs, batch['emotion'])
        else:
            loss = ce_loss


        self.val_loss.append(loss.item())
        if self.is_tune is False:
            self.val_pred.append(outputs)
            self.val_y_true.append(batch['emotion'])

        self.log('val/acc', self.acc(outputs, batch['emotion']), on_step=False, on_epoch=True)
        self.log('val/f1', self.f1(outputs, batch['emotion']), on_step=False, on_epoch=True)
        self.log('val/ce_loss', ce_loss, on_step=True, on_epoch=False)
        self.log('val/focal_loss', loss, on_step=True, on_epoch=False)
        return {'loss': loss, 'preds': outputs, 'target': batch['emotion']}

    def on_validation_epoch_end(self):
        self.log('val/mean_focal_loss', np.array(self.val_loss).mean())
        self.val_loss = []
        if self.is_tune is False:
            preds = torch.cat([i for i in self.val_pred])
            targets = torch.cat([i for i in self.val_y_true])
            self.confusion_matrix.draw_confusion_matrix(preds, targets, self.current_epoch, self.logger,
                                                        "val/Confusion matrix")
            self.val_pred = []
            self.val_y_true = []

    def configure_optimizers(self):
        all_params = list(self.parameters())

        optimizer = torch.optim.Adam([
            {"params": all_params},
        ],
            lr=self.lr
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in [100, 200, 300, 400, 650, 800]],
            gamma=(1. / 3.)
        )
        return [optimizer], [scheduler]

    @staticmethod
    def getParameters(model, rec=None):
        rec = rec or []
        parameters = []
        for index, layer in enumerate(model.children()):
            paramdict = {'params': layer.parameters(), "_name": '_'.join(map(str, rec[:] + [index]))}
            parameters.append(paramdict)
        return parameters

    @classmethod
    def get_recourcive_params(cls, model, recourcive_map: list[int]):
        params = [None]
        next_model = model
        last_i = 0
        target_params = None
        for index, i in enumerate([] + recourcive_map):
            nested_params = cls.getParameters(next_model, recourcive_map[:index])
            params = params[:last_i] + nested_params + params[last_i + 1:]
            if index == len(recourcive_map) - 1:
                nested_params[i] |= {"lr": 1e-4}
                target_params = nested_params[i]
            else:
                next_model = list(next_model.children())[i]
            last_i = i
        return params, target_params