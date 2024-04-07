from os.path import join, dirname, split
import os
from datetime import datetime
from pprint import pprint

import torch
import torchmetrics
import numpy as np
import lightning as L
import torch.nn.functional as F
from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import BackboneFinetuning
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from config.constants import ROOT_DIR, PADDING_SEC
from utils.data import EmotionDataset
from model import Wav2Vec2Classifier


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'




def group_wise_lr(model, group_lr_conf: dict, path=""):
    """
    Refer https://pytorch.org/docs/master/optim.html#per-parameter-options


    torch.optim.SGD([
        {'params': model.base.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], lr=1e-2, momentum=0.9)


    to


    cfg = {"classifier": {"lr": 1e-3},
           "lr":1e-2, "momentum"=0.9}
    confs, names = group_wise_lr(model, cfg)
    torch.optim.SGD([confs], lr=1e-2, momentum=0.9)



    :param model:
    :param group_lr_conf:
    :return:
    """
    assert type(group_lr_conf) == dict
    confs = []
    nms = []
    for kl, vl in group_lr_conf.items():
        assert type(kl) == str
        assert type(vl) == dict or type(vl) == float or type(vl) == int

        if type(vl) == dict:
            assert hasattr(model, kl)
            cfs, names = group_wise_lr(getattr(model, kl), vl, path=path + kl + ".")
            confs.extend(cfs)
            names = list(map(lambda n: kl + "." + n, names))
            nms.extend(names)

    primitives = {kk: vk for kk, vk in group_lr_conf.items() if type(vk) == float or type(vk) == int}
    remaining_params = [(k, p) for k, p in model.named_parameters() if k not in nms]
    if len(remaining_params) > 0:
        names, params = zip(*remaining_params)
        conf = dict(params=params, **primitives)
        confs.append(conf)
        nms.extend(names)

    plen = sum([len(list(c["params"])) for c in confs])
    assert len(list(model.parameters())) == plen
    assert set(list(zip(*model.named_parameters()))[0]) == set(nms)
    assert plen == len(nms)
    if path == "":
        for c in confs:
            c["params"] = (n for n in c["params"])
    return confs, nms


class LitModule(L.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = Wav2Vec2Classifier(num_classes, PADDING_SEC)

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
        # pprint([dict(i) for i in list(self.parameters())])
        # print(list(list(list(self.children())[0].children())[3].children()), sep = "\n\n")

        # print(self.getParameters(self))




        params, target_params = self.get_recourcive_params(self, [0, 3, 0])
        all_params = list(set(self.parameters()) - set(target_params['params']))
        # all_params = self.parameters()
        # print(params)
        # optimizer = torch.optim.Adam(params, lr=1e-4)

        optimizer = torch.optim.Adam([
            {"params": all_params},
            target_params,
        ], lr=1e-3)

        # optimizer2 = torch.optim.Adam([
        #     target_params,
        #
        # ], lr=1e-2)

        # optimizer.add_param_group({'params': self.model.feature_resizer.parameters(), "lr": 1e-5})
        # print(len( optimizer.param_groups))
        # for g in optimizer.param_groups:
        #     print("--", g)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i * 2 for i in [15, 35, 55, 80, 100, 120, 140, 160, 180, 200, 220]],
            gamma=(1./3.)
        )
        return [optimizer], [scheduler]

    @staticmethod
    def getParameters(model, rec=None):
        rec = rec or []
        # getWidthConv2D = lambda layer: layer.out_channels
        parameters = []
        # print(list(list(model.children())[0].children()), sep = "\n\n")
        for index, layer in enumerate(model.children()):
            paramdict = {'params': layer.parameters(), "_name": '_'.join(map(str, rec[:] + [index]))}
            # if (isinstance(layer, nn.Conv2d)):
            #     print(paramdict)
            #     paramdict['lr'] = getWidthConv2D(layer) * 0.1 # Specify learning rate for Conv2D here
            parameters.append(paramdict)
        return parameters

    @classmethod
    def get_recourcive_params(cls, model, recourcive_map: list[int]):
        params = [None]
        next_model = model
        last_i = 0
        target_params = None
        for index, i in enumerate( [] + recourcive_map ):
            # print(i, len(list(next_model.children())))
            nested_params = cls.getParameters(next_model, recourcive_map[:index])
            # print(params[:last_i], params[last_i+1:])
            params = params[:last_i] + nested_params + params[last_i+1:]
            if index == len(recourcive_map) -1:
                nested_params[i] |= {"lr" : 1e-4}
                target_params =  nested_params[i]
                # print(next_model, nested_params)
            else:
                next_model = list(next_model.children())[i]
            last_i = i
        # print("\n----", *[i| {"params": id(i["params"])} for i in params], sep='\n')
        return params, target_params




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



class FCNetworkBackboneFinetuning(BackboneFinetuning):
    pass



def main():

    trainer = None

    dataset = EmotionDataset(annotation= join(ROOT_DIR, 'data', 'dataset', 'annotations.json'), padding_sec=PADDING_SEC)

    train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                          test_size=0.15,
                                          random_state=42,
                                          shuffle=True,
                                          stratify=dataset.emotion_labels)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=collate_fn, num_workers=15)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn, num_workers=15)

    print("classifier_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_{epoch:02d}")

    # lr_logger = LearningRateLogger()  # add lr logger callback to trainer …
    # trainer = Trainer(callbacks=[]) # train model as usual


    checkpoint_callback = ModelCheckpoint(
        dirpath= join(ROOT_DIR, "weights",  'checkpoints',  datetime.now().strftime("%Y%m%d-%H%M%S")),
        filename="classifier_" +  datetime.now().strftime("%Y%m%d-%H%M%S") + "_{epoch:02d}",
        every_n_epochs=10,
        save_top_k=-1,
    )
    # print(ModelCheckpoint.dirpath)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = LitModule(len(dataset.emotions))
    trainer = L.Trainer(accelerator='gpu',
                        devices=1,
                        max_epochs=600,
                        callbacks=[checkpoint_callback, lr_monitor],
                        default_root_dir=join(ROOT_DIR, 'logs')
                        )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
