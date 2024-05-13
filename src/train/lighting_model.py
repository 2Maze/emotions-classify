import lightning as L
import numpy as np
import torch.nn.functional as F
import torchmetrics
import torch
from vit_pytorch import ViT
from torch import Tensor

from metrics.confusion_matrix import CreateConfMatrix
from models import Wav2Vec2FnClassifier, Wav2Vec2CnnClassifier, SpectrogramCnnClassifier, TransformerClassifier
from config.constants import ROOT_DIR, PADDING_SEC


def nn_model_choice(config: dict):
    if config['model'] == 'Wav2Vec2FnClassifier':
        return Wav2Vec2FnClassifier
    elif config['model'] == 'Wav2Vec2CnnClassifier':
        return Wav2Vec2CnnClassifier
    elif config['model'] == 'SpectrogramCnnClassifier':
        return SpectrogramCnnClassifier
    elif config['model'] == 'TransformerClassifier':
        return TransformerClassifier


class LitModule(L.LightningModule):
    def __init__(
            self,
            emotions_count: int,
            states_count: int,
            # learning_rate: float = 1e-3,
            # conv_learning_rate: float = 1e-3,
            # conv_h_count: int = 0,
            # conv_w_count: int = 0,
            # padding_sec: int = 0,
            # layer_1_size: int = 1024,
            # layer_2_size: int = 512,
            # h_mean_enable: bool = True,
            # w_mean_enable: bool = True,
            config: dict,
            # lables: list[str] = None,
            # is_tune: bool = False,
            dataset: list = None,
            # gamma: float = 2.0
    ):
        super().__init__()
        if (config['type'] == 'train'):
            self.save_hyperparameters()
        # self.model = SpectrogramCnnClassifier(
        #     num_classes,
        #     dataset=dataset,
        #     # padding_sec,
        #     # conv_h_count=conv_h_count,
        #     # conv_w_count=conv_w_count,
        #     # layer_1_size=layer_1_size,
        #     # layer_2_size=layer_2_size,
        #     config=config
        # )
        # efficient_transformer = Linformer(
        #     dim=128,
        #     seq_len=49 + 1,  # 7x7 patches + 1 cls-token
        #     depth=12,
        #     heads=8,
        #     k=64
        # )


#         self.model = ViT(
#             # transformer=efficient_transformer,
#     image_size = 512,
#     patch_size = 64,
#     num_classes = 7,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1,
#             channels=1,
# )

        self.model = nn_model_choice(config)(
            emotions_count,
            states_count,
            dataset=dataset,
            config=config,
        )

        self.config = config
        self.is_tune = config.get("tune", False)
        self.emotions_count = emotions_count

        self.lr = config['learn_params']['lr']['base_lr']
        self.conv_lr = config['learn_params']['lr'].get('conv_lr', 0)

        self.val_loss = []
        self.st_val_loss = []

        self.val_pred = []
        self.val_y_true = []
        self.val_pred_st = []
        self.val_y_true_st = []

        self.train_y_pred = []
        self.train_y_true = []
        self.train_y_pred_st = []
        self.train_y_true_st = []

        self.acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=emotions_count)
        self.f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=emotions_count)

        self.st_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=states_count)
        self.st_f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=states_count)

        self.emotion_confusion_matrix = CreateConfMatrix(emotions_count, list(dataset.emotions), self.logger)
        self.states_confusion_matrix = CreateConfMatrix(states_count, list(dataset.states), self.logger)

        self.class_weights = []
        class_counts = np.bincount([i['emotion'] for i in dataset])
        emotions_count = len(class_counts)
        total_samples = len(dataset)
        for count in class_counts:
            # print(count,  total_samples)
            if count != 0:
                weight = 1 / (count / total_samples)
            else:
                weight = 0
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
        # print("training_step", outputs)

        if isinstance(outputs, Tensor):
            emotion_outputs = outputs
            state_outputs = Tensor([])
        else:
            emotion_outputs = outputs[0]
            state_outputs = outputs[1]


        emotion_ce_loss = F.cross_entropy(emotion_outputs, batch['emotion'])
        state_ce_loss = F.cross_entropy(state_outputs, batch['state'])

        main_loss = emotion_ce_loss + state_ce_loss
        # if self.focal_loss is not None:
        #     loss = self.focal_loss(emotion_outputs, batch['emotion'])
        # else:
        #     loss = emotion_ce_loss
        self.log('train/ce_em_loss', emotion_ce_loss, on_step=True, on_epoch=False)
        self.log('train/ce_st_loss', state_ce_loss, on_step=True, on_epoch=False)
        # self.log('train/focal_loss', loss, on_step=True, on_epoch=False)
        if self.is_tune is False:
            self.train_y_pred.append(emotion_outputs)
            self.train_y_true.append(batch['emotion'])
            self.train_y_pred_st.append(state_outputs)
            self.train_y_true_st.append(batch['state'])

        return main_loss

    def on_train_epoch_end(self):
        if self.is_tune is False:

            preds = torch.cat([i for i in self.train_y_pred])
            targets = torch.cat([i for i in self.train_y_true])
            preds_st = torch.cat([i for i in self.train_y_pred_st])
            targets_st = torch.cat([i for i in self.train_y_true_st])

            self.emotion_confusion_matrix.draw_confusion_matrix(preds, targets, self.current_epoch, self.logger,
                                                        "train/emotion Confusion matrix")
            self.states_confusion_matrix.draw_confusion_matrix(preds_st, targets_st, self.current_epoch, self.logger,
                                                                "train/state Confusion matrix")

            self.train_y_pred = []
            self.train_y_true = []
            self.train_y_pred_st = []
            self.train_y_true_st = []

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['array'])
        # print("validation_step", outputs)

        if isinstance(outputs, Tensor):
            emotion_outputs = outputs
            state_outputs = Tensor([])
        else:
            emotion_outputs = outputs[0]
            state_outputs = outputs[1]


        emotion_ce_loss = F.cross_entropy(emotion_outputs, batch['emotion'])
        state_ce_loss = F.cross_entropy(state_outputs, batch['state'])

        self.val_loss.append(emotion_ce_loss.item())
        self.st_val_loss.append(state_ce_loss.item())
        if self.is_tune is False:
            self.val_pred.append(emotion_outputs)
            self.val_y_true.append(batch['emotion'])
            self.val_pred_st.append(state_outputs)
            self.val_y_true_st.append(batch['state'])


        main_loss = emotion_ce_loss + state_ce_loss

        self.log('val/em_acc', self.acc(emotion_outputs, batch['emotion']), on_step=False, on_epoch=True)
        self.log('val/em_f1', self.f1(emotion_outputs, batch['emotion']), on_step=False, on_epoch=True)
        self.log('val/ce_em_loss', emotion_ce_loss, on_step=True, on_epoch=False)

        self.log('val/st_acc', self.st_acc(state_outputs, batch['state']), on_step=False, on_epoch=True)
        self.log('val/st_f1', self.st_f1(state_outputs, batch['state']), on_step=False, on_epoch=True)
        self.log('val/ce_st_loss', state_ce_loss, on_step=True, on_epoch=False)
        # self.log('val/focal_loss', loss, on_step=True, on_epoch=False)
        return {
            'loss': main_loss,
            'em_loss': emotion_ce_loss, "st_loss": state_ce_loss,
            'em_preds': emotion_outputs, 'em_target': batch['emotion'],
            'st_preds': state_outputs, 'st_target': batch['state'],
        }

    def on_validation_epoch_end(self):
        self.log('val/mean_em_loss', np.array(self.val_loss).mean())
        self.log('val/mean_st_loss', np.array(self.st_val_loss).mean())
        self.val_loss = []
        self.st_val_loss = []
        if self.is_tune is False:

            preds = torch.cat([i for i in self.val_pred])
            targets = torch.cat([i for i in self.val_y_true])
            st_preds = torch.cat([i for i in self.val_pred_st])
            st_targets = torch.cat([i for i in self.val_y_true_st])

            self.emotion_confusion_matrix.draw_confusion_matrix(preds, targets, self.current_epoch, self.logger,
                                                        "val/emotion Confusion matrix")
            self.states_confusion_matrix.draw_confusion_matrix(st_preds, st_targets, self.current_epoch, self.logger,
                                                                "val/state Confusion matrix")

            self.val_pred = []
            self.val_y_true = []
            self.val_pred_st = []
            self.val_y_true_st = []

    def configure_optimizers(self):
        all_params = list(self.parameters())

        optimizer = torch.optim.Adam([
            {"params": all_params},
        ],
            lr=self.lr
        )
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in [ 300,   650, 800]],
            gamma=0.1
        ),
        }
        return [optimizer], [lr_scheduler]

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
