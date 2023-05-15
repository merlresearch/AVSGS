# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import torch
import torchvision
from easydict import EasyDict as edict

from .networks import (
    AudioVisual5layerUNet,
    AudioVisual7layerUNet,
    GraphNet_0,
    GraphNet_1,
    GraphNet_2,
    GraphNet_3,
    Resnet18,
    weights_init,
)


class ModelBuilder:
    # builder for visual stream
    def build_visual(self, pool_type="avgpool", input_channel=3, fc_out=512, weights=""):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        if pool_type == "conv1x1":  # if use conv1x1, use conv1x1 + fc to reduce dimension to 512 feature vector
            net = Resnet18(
                original_resnet, pool_type=pool_type, input_channel=3, with_fc=True, fc_in=6272, fc_out=fc_out
            )
        else:
            net = Resnet18(original_resnet, pool_type=pool_type)

        if len(weights) > 0:
            print("Loading weights for visual stream")
            net.load_state_dict(torch.load(weights))
        return net

    # builder for audio stream
    def build_unet(self, unet_num_layers=7, ngf=64, input_nc=1, output_nc=1, weights=""):
        if unet_num_layers == 7:
            net = AudioVisual7layerUNet(ngf, input_nc, output_nc)
        elif unet_num_layers == 5:
            net = AudioVisual5layerUNet(ngf, input_nc, output_nc)

        net.apply(weights_init)

        if len(weights) > 0:
            print("Loading weights for UNet")
            net.load_state_dict(torch.load(weights))
        return net

    # builder for audio classifier stream
    def build_classifier(self, pool_type="avgpool", num_of_classes=15, input_channel=1, weights=""):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18(
            original_resnet,
            pool_type=pool_type,
            input_channel=input_channel,
            with_fc=True,
            fc_in=512,
            fc_out=num_of_classes,
        )

        if len(weights) > 0:
            print("Loading weights for audio classifier")
            net.load_state_dict(torch.load(weights))
        return net

    # builder for graph encoder network
    def build_graph_encoder(
        self,
        feat_dim=2048,
        hidden_act="relu",
        fin_graph_rep=256,
        pooling_ratio=0.5,
        nos_classes=16,
        gnet_type=3,
        heads=4,
        weights="",
    ):
        self.args = edict(
            {
                "nhid": fin_graph_rep,
                "pooling_ratio": 0.5,
                "num_features": feat_dim,  # * + audio_rep,
                "heads": heads,
                "hidden_act": hidden_act,
                "nos_classes": nos_classes,
            }
        )
        if gnet_type == 3:
            self.gh_encoder = GraphNet_3(self.args)
        elif gnet_type == 2:
            self.gh_encoder = GraphNet_2(self.args)
        elif gnet_type == 1:
            self.gh_encoder = GraphNet_1(self.args)
        else:
            self.gh_encoder = GraphNet_0(self.args)

        if len(weights) > 0:
            print("Loading weights for graph network")
            self.gh_encoder.load_state_dict(torch.load(weights))

        return self.gh_encoder
