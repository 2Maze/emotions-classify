from os.path import join, dirname
from collections import OrderedDict

import torchvision.models as models
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers import AutoImageProcessor, ViTModel
from vit_pytorch import ViT


from config.constants import ROOT_DIR, PADDING_SEC


class TransformerClassifier(nn.Module):

    def __init__(
            self,
            num_classes,
            # padding_sec,
            config: dict | None = None,
            **k_,
    ):
        super(TransformerClassifier, self).__init__()
        self.padding_sec = config['padding_sec']
        self.padding_sec_w = 50 *  self.padding_sec - 1

        self.config = config

        self.layer_1_size = config['layer_1_size']
        self.layer_2_size = config['layer_2_size']

        self.model = ViT(
            # transformer=efficient_transformer,
            image_size=512,
            patch_size=config['patch_transformer_size'],
            num_classes=7,
            dim=config['layer_1_size'],
            depth=config['transformer_depth'],
            heads=config['transformer_attantion_head_count'],
            mlp_dim=config['layer_2_size'],
            dropout=0.1,
            emb_dropout=0.1,
            channels=1,
        )

        # configuration = Wav2Vec2Config()

        # self.image_processor = AutoImageProcessor.from_pretrained(
        #     "google/vit-base-patch16-224-in21k",
        #     cache_dir=join(ROOT_DIR, "weights", "loaded_weights",)
        # )
        #
        # self.model = ViTModel.from_pretrained(
        #     "google/vit-base-patch16-224-in21k",
        #     cache_dir=join(ROOT_DIR, "weights", "loaded_weights",)
        # ).to('cuda:0')

        # self.classifier = nn.Sequential(
        #     nn.Linear( 768,
        #         self.layer_1_size,
        #         bias=True
        #     ),
        #     nn.BatchNorm1d(1 * self.layer_1_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1 * self.layer_1_size, self.layer_2_size, bias=True),
        #     nn.BatchNorm1d(self.layer_2_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.layer_2_size, num_classes, bias=True)
        # )

        # self.efficientnet_model = models.efficientnet_b0(weights=False)
        #
        # self.efficientnet_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
        #                                                    bias=False)
        # self.efficientnet_model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

        # for param in self.embedding_model.wav2vec2.parameters():
        #     param.requires_grad = False

        # self.classifier = self.efficientnet_model

    def forward(
            self,
            X: torch.Tensor,
    ):
        # image = torch.nn.functional.normalize(X.view(-1, 1, 401, 512), dim=0)
        # image = torch.cat((image, image, image), dim=1)
        # inputs = self.image_processor(image, return_tensors="pt", do_rescale=False, ).to('cuda:0')
        # print(X.size())
        inputs = self.model(X.view(-1, 1, 512, 512))
        # logits = self.classifier(inputs)
        # logits = torch.softmax(inputs, dim=1)
        return inputs
