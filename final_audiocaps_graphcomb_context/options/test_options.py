# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

from .base_options import BaseOptions


# test by mix and separate two videos
class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--video1_name", type=str, default="")
        self.parser.add_argument("--video2_name", type=str, default="")
        self.parser.add_argument("--output_dir_root", type=str, default="output")
        self.parser.add_argument(
            "--hop_size",
            default=0.05,
            type=float,
            help="the hop length to perform audio separation in a sliding window approach",
        )
        self.parser.add_argument(
            "--subtract_mean", default=True, type=bool, help="subtract channelwise mean from input image"
        )
        self.parser.add_argument(
            "--preserve_ratio",
            default=False,
            type=bool,
            help="whether boudingbox aspect ratio should be preserved when loading",
        )
        self.parser.add_argument(
            "--enable_data_augmentation", type=bool, default=False, help="whether to augment input audio/image"
        )
        self.parser.add_argument(
            "--spectrogram_type",
            type=str,
            default="magonly",
            choices=("complex", "magonly"),
            help="whether to use magonly or complex spectrogram",
        )
        self.parser.add_argument("--with_discriminator", action="store_true", help="whether to use discriminator")
        self.parser.add_argument("--visualize_spectrogram", action="store_true", help="whether to use discriminator")

        # model specification
        self.parser.add_argument(
            "--visual_pool", type=str, default="maxpool", help="avg or max pool for visual stream feature"
        )
        self.parser.add_argument(
            "--classifier_pool", type=str, default="maxpool", help="avg or max pool for classifier stream feature"
        )
        self.parser.add_argument("--weights_visual", type=str, default="", help="weights for visual stream")
        self.parser.add_argument("--weights_unet", type=str, default="", help="weights for unet")
        self.parser.add_argument("--weights_classifier", type=str, default="", help="weights for audio classifier")
        self.parser.add_argument("--weights_graph", type=str, default="", help="weights of graph encoder")
        self.parser.add_argument("--weights_map_net", type=str, default="", help="weights of map net")
        self.parser.add_argument("--weights_rnn", type=str, default="", help="weights of the rnn")
        self.parser.add_argument(
            "--weights_rnn_classif", type=str, default="", help="weights of the  classifier sitting on the rnn output"
        )
        self.parser.add_argument("--unet_num_layers", type=int, default=7, choices=(5, 7), help="unet number of layers")
        self.parser.add_argument("--unet_ngf", type=int, default=64, help="unet base channel dimension")
        self.parser.add_argument("--unet_input_nc", type=int, default=1, help="input spectrogram number of channels")
        self.parser.add_argument("--unet_output_nc", type=int, default=1, help="output spectrogram number of channels")
        self.parser.add_argument("--number_of_classes", default=14, type=int, help="number of classes")
        self.parser.add_argument(
            "--with_silence_category", action="store_true", help="whether to augment input audio/image"
        )
        self.parser.add_argument("--weighted_loss", action="store_true", help="weighted loss")
        self.parser.add_argument(
            "--binary_mask", action="store_true", help="whether use binary mask, ratio mask is used otherwise"
        )
        self.parser.add_argument("--full_frame", action="store_true", help="pass full frame instead of object regions")
        self.parser.add_argument("--mask_thresh", default=0.5, type=float, help="mask threshold for binary mask")
        self.parser.add_argument("--feat_dim", default=2048, type=int, help="dimension of node features of graph")
        self.parser.add_argument("--graph_loss_weight", default=0.5, type=float, help="weight for classifier loss")
        self.parser.add_argument("--graph_enc_dim", default=256, type=int, help="dimension of encoded graph")
        self.parser.add_argument("--pooling_ratio", default=0.5, type=float, help="the pooling ratio")
        self.parser.add_argument("--gnet_type", default=3, type=int, help="which graph network to use")
        self.parser.add_argument("--hidden_act", default="relu", type=str, help="type of graph activation to use")
        self.parser.add_argument(
            "--conf_thresh",
            default=0.4,
            type=float,
            help="the minimum objectness confidence threshold for a context node to be associated with a sounding object",
        )
        self.parser.add_argument(
            "--nos_cntxt",
            default=20,
            type=int,
            help="the maximum number of context nodes to be associated with a sounding object",
        )
        self.parser.add_argument(
            "--iou_thresh",
            default=0.1,
            type=float,
            help="the minimum iou required for a person node to be associated with an instrument",
        )
        self.parser.add_argument("--log_freq", type=bool, default=True, help="whether use log-scale frequency")
        self.parser.add_argument(
            "--with_frame_feature", action="store_true", help="whether also use frame-level visual feature"
        )
        self.parser.add_argument(
            "--with_additional_scene_image", action="store_true", help="whether append an extra scene image"
        )
        self.parser.add_argument(
            "--num_of_object_detections_to_use", type=int, default=1, help="num of predictions to avg"
        )
        # include test related hyper parameters here
        self.parser.add_argument("--lr_graph_net", type=float, default=0.0001, help="learning rate for graph encoder")
        self.mode = "test"
