# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import EdgeConv, GATConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, input):
        return gelu(input)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return swish(input)


def MLP(channels, batch_norm=True):
    return Seq(*[Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i])) for i in range(1, len(channels))])


act_fn = {"gelu": gelu, "relu": F.relu, "swish": swish}
act_module = {"gelu": GELU(), "relu": nn.ReLU(), "swish": Swish()}


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))

    if Relu:
        model.append(nn.ReLU())

    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)


class GraphNet_0(torch.nn.Module):
    def __init__(self, args):
        super(GraphNet_0, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.heads = args.heads
        self.hidden_act = args.hidden_act
        self.num_classes = args.nos_classes

        self.node_conv = GATConv(self.num_features, self.nhid // self.heads, heads=self.heads)

    def forward(self, data):
        x, edge_index, batch = data  # *.x, data.edge_index, data.batch
        x = act_fn[self.hidden_act](self.node_conv(x, edge_index))
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        return x


class GraphNet_1(torch.nn.Module):
    def __init__(self, args):
        super(GraphNet_1, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.heads = args.heads
        self.hidden_act = args.hidden_act
        self.num_classes = args.nos_classes

        self.edge_conv1 = EdgeConv(MLP([2 * self.num_features, 2 * self.nhid, self.nhid]), aggr="max")
        self.edge_conv2 = EdgeConv(MLP([2 * self.nhid, self.nhid]), aggr="max")

    def forward(self, data):
        x, edge_index, batch = data  # *.x, data.edge_index, data.batch
        x1 = act_fn[self.hidden_act](self.edge_conv1(x, edge_index))
        x2 = act_fn[self.hidden_act](self.edge_conv2(x1, edge_index))

        x_fuse = x1 + x2
        x_out = torch.cat([gmp(x_fuse, batch), gap(x_fuse, batch)], dim=1)

        return x_out


class GraphNet_2(torch.nn.Module):
    def __init__(self, args):
        super(GraphNet_2, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.heads = args.heads
        self.hidden_act = args.hidden_act
        self.num_classes = args.nos_classes

        self.node_conv = GATConv(self.num_features, self.nhid // self.heads, heads=self.heads)
        self.edge_conv = EdgeConv(MLP([2 * self.num_features, 2 * self.nhid, self.nhid]), aggr="max")

    def forward(self, data):
        x, edge_index, batch = data  # *.x, data.edge_index, data.batch
        x_node = act_fn[self.hidden_act](self.node_conv(x, edge_index))
        x_edge = act_fn[self.hidden_act](self.edge_conv(x, edge_index))

        x_out = torch.cat([gap(x_node, batch), gap(x_edge, batch)], dim=1)

        return x_out


class GraphNet_3(torch.nn.Module):
    def __init__(self, args):
        super(GraphNet_3, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio
        self.heads = args.heads
        self.hidden_act = args.hidden_act
        self.num_classes = args.nos_classes

        self.node_conv = GATConv(self.num_features, self.nhid // self.heads, heads=self.heads)
        self.edge_conv = EdgeConv(MLP([2 * self.nhid, self.nhid]), aggr="max")

    def forward(self, data):
        x, edge_index, batch = data  # *.x, data.edge_index, data.batch
        x_node = act_fn[self.hidden_act](self.node_conv(x, edge_index))
        x_edge = act_fn[self.hidden_act](self.edge_conv(x_node, edge_index))

        x_node = torch.cat([gmp(x_node, batch), gap(x_node, batch)], dim=1)
        x_edge = torch.cat([gmp(x_edge, batch), gap(x_edge, batch)], dim=1)
        x_out = x_node + x_edge

        return x_out  # *, self.mlp_layers(x_out)


class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type="maxpool", input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        # customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers)  # features before pooling

        if pool_type == "conv1x1":
            self.conv1x1 = create_conv(512, 128, 1, 0)
            self.conv1x1.apply(weights_init)

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)

        if self.pool_type == "avgpool":
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == "maxpool":
            x = F.adaptive_max_pool2d(x, 1)
        elif self.pool_type == "conv1x1":
            x = self.conv1x1(x)
        else:
            return x  # no pooling and conv1x1, directly return the feature map

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            if self.pool_type == "conv1x1":
                x = x.view(x.size(0), -1, 1, 1)  # expand dimension if using conv1x1 + fc to reduce dimension
            return x
        else:
            return x


class AudioVisual7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual7layerUNet, self).__init__()

        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(
            ngf * 2, output_nc, True
        )  # outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        # *print('Encoded audio shape: ' + str(audio_conv7feature.size()) + ' encoded video shape: ' + str(visual_feat.size()))
        audioVisual_feature = torch.cat((visual_feat, audio_conv7feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        return mask_prediction


class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual5layerUNet, self).__init__()

        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(
            ngf * 2, output_nc, True
        )  # outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[2], audio_conv5feature.shape[3])
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return mask_prediction
