from os.path import join, dirname
from collections import OrderedDict

import torch
import torch.nn as nn
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers import Wav2Vec2Model, Wav2Vec2Config

from config.constants import ROOT_DIR, PADDING_SEC


class Wav2Vec2Classifier(nn.Module):

    def __init__(
            self,
            num_classes,
            # padding_sec,
            # conv_h_count: int = 0,
            # conv_w_count: int = 0,
            # layer_1_size: int = 1024,
            # layer_2_size: int = 512,
            config: dict | None,
    ):
        super(Wav2Vec2Classifier, self).__init__()
        self.padding_sec = config['padding_sec']
        self.padding_sec_w = 50 * self.padding_sec - 1

        self.conv_h_count = config['conv_h_count']
        self.conv_w_count = config['conv_w_count']
        self.layer_1_size = config['layer_1_size']
        self.layer_2_size = config['layer_2_size']
        self.config = config

        # configuration = Wav2Vec2Config()
        self.embedding_model: Wav2Vec2ForCTC = AutoModelForCTC.from_pretrained(
            "Eyvaz/wav2vec2-base-russian-demo-kaggle",
            cache_dir=join(ROOT_DIR, "weights", "loaded_weights", ),
            # config=configuration,
        )
        processor = AutoProcessor.from_pretrained(
            "Eyvaz/wav2vec2-base-russian-demo-kaggle",
            cache_dir=join(ROOT_DIR, "weights", "loaded_weights", ),
        )
        self.feature_extractor = self.embedding_model.wav2vec2.feature_extractor
        self.feature_projection = self.embedding_model.wav2vec2.feature_projection
        self.feature_post_projection = nn.Sequential(
            self.embedding_model.wav2vec2.encoder.pos_conv_embed,
            self.embedding_model.wav2vec2.encoder.layer_norm,
        )

        if (self.conv_h_count > 0):
            self.feature_resizer1 = nn.Sequential(
                OrderedDict([
                    ("1d_conv", nn.Conv1d(self.padding_sec_w, self.conv_h_count, 1)),
                    ("bath_after_1d_conv", nn.BatchNorm1d(self.conv_h_count)),
                    ("relu_after_1d_conv", nn.ReLU(inplace=True)),
                ]))
        else:
            self.feature_resizer1 = nn.Sequential()

        if self.conv_w_count > 0:
            self.feature_resizer2 = nn.Sequential(
                OrderedDict([
                    ("1d_conv", nn.Conv1d(512, self.conv_w_count, 1)),
                    ("bath_after_1d_conv", nn.BatchNorm1d(self.conv_w_count)),
                    ("relu_after_1d_conv", nn.ReLU(inplace=True)),
                ]))
        else:
            self.feature_resizer2 = nn.Sequential()

        # bundle = torchaudio.pipelines.WAV2VEC2_BASE
        # self.feature_extractor = bundle.get_model(
        #     dl_kwargs={
        #         "file_name": join(ROOT_DIR, "weights", "loaded_weights", "wav2vec2_fairseq_base_ls960.pth")
        #     }
        # )

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        # for param in self.feature_projection.parameters():
        #     param.requires_grad = False
        # for param in  self.feature_post_projection.parameters():
        #     param.requires_grad = False
        for param in self.embedding_model.wav2vec2.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(
                self.padding_sec_w * (1 + self.conv_w_count)
                + 512 * (1 + self.conv_h_count), 1 * self.layer_1_size,
                bias=True
            ),
            nn.BatchNorm1d(1 * self.layer_1_size),
            nn.ReLU(inplace=True),
            nn.Linear(1 * self.layer_1_size, self.layer_2_size, bias=True),
            # nn.Linear(self.padding_sec_w * 4, 1 *  1024, bias=True),
            # nn.Linear(867 * 1, 1 * 1024, bias=True),
            # nn.BatchNorm1d(1 * 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1 * 1024, 512, bias=True),
            nn.BatchNorm1d(self.layer_2_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.layer_2_size, num_classes, bias=True)
        )
        # print(self.classifier[0])

    def forward(
            self,
            X: torch.Tensor,
            mask_time_indices: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
    ):
        features = self.get_embeddings(X, mask_time_indices=mask_time_indices, attention_mask=attention_mask)
        # print("-----features", features.size(),
        #       self.conv_h_count,
        #     self.conv_w_count,
        #     self.layer_1_size,
        #     self.layer_2_size,
        #     self.config,
        #       )
        logits = self.classifier(features)
        logits = torch.softmax(logits, dim=1)
        return logits

    def get_embeddings(
            self,
            input_values: torch.Tensor,
            mask_time_indices: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
    ):
        # print("\n----------X",input_values.size())
        # embeddings = self.feature_extractor(X)[0].mean(axis=1)
        # embeddings = self.feature_extractor(X)
        # res = nn.functional.normalize(embeddings)
        # return res

        # extract_features = self.feature_extractor(input_values)
        # extract_features = extract_features.transpose(1, 2)
        #
        # if attention_mask is not None:
        #     # compute reduced attention_mask corresponding to feature vectors
        #     attention_mask = self._get_feature_vector_attention_mask(
        #         extract_features.shape[1], attention_mask, add_adapter=False
        #     )
        #
        # hidden_states, extract_features = self.feature_projection(extract_features)
        # hidden_states = self.embedding_model.wav2vec2._mask_hidden_states(
        #     hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        # )
        hidden_states = self.embedding_model.wav2vec2(input_values).extract_features
        # print(hidden_states.size())
        # hidden_states = self.feature_post_projection(hidden_states)

        resize_tensors = []

        if self.conv_h_count > 0:
            # print(hidden_states.size())
            resized_features1 = self.feature_resizer1(hidden_states.view(-1, self.padding_sec_w * 1, 512))
            resized_features1 = resized_features1.view(-1, 512 * self.conv_h_count)
            resize_tensors.append(resized_features1)
        if self.conv_w_count > 0:
            # print(hidden_states.size())
            resized_features2 = self.feature_resizer2(hidden_states.view(-1, 512 * 1, self.padding_sec_w))
            resized_features2 = resized_features2.view(-1, self.padding_sec_w * self.conv_w_count)
            resize_tensors.append(resized_features2)

        # resized_features = torch.cat((resized_features1, resized_features2), 1)

        # resized_features = resized_features2
        # resized_features = hidden_states.mean(axis=1)
        resized_features = torch.cat((hidden_states.mean(axis=1), hidden_states.mean(axis=2), *resize_tensors), 1)
        resized_features = nn.functional.normalize(resized_features)

        # res_ =
        # print("final_res", res_, type(res_))
        # print(res_.size())
        # exit(-1)

        # print("resized_features", hidden_states.size(), resized_features.size())

        # print(mask_time_indices.size(), attention_mask.size())

        # # summary()
        # # print("embeddings", hidden_states.size(), self.feature_resizer)
        # hidden_states = hidden_states.view(1, self.padding_sec_w, 768, -1)
        # # print("embeddings", hidden_states.size(), self.feature_resizer)
        #
        # resized_features = self.feature_resizer(hidden_states)
        # resized_features = resized_features.view(-1, 4 * 768 )
        # # print("resized_features", resized_features.size(), 16 * 768)
        # # exit(0)
        # exit(-1)
        return resized_features
