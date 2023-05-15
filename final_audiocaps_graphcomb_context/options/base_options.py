# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import argparse
import os

import torch
from utils import utils


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--data_path", default="/your_data_root/Audiocaps/solo/", help="path to frame/audio/detections"
        )
        self.parser.add_argument("--hdf5_path", default="/your_root/hdf5/Audiocaps/soloduet")
        self.parser.add_argument("--scene_path", default="/your_root/hdf5/ADE.h5", help="path to scene images")
        self.parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        self.parser.add_argument(
            "--name",
            type=str,
            default="audiocaps_noclassif_graphcomb_noaug",
            help="name of the experiment. It decides where to store models",
        )
        self.parser.add_argument("--checkpoints_dir", type=str, default="checkpoints/", help="models are saved here")
        self.parser.add_argument("--model", type=str, default="ASIW", help="chooses how datasets are loaded.")
        self.parser.add_argument("--batchSize", type=int, default=10, help="input batch size")
        self.parser.add_argument("--nThreads", default=16, type=int, help="# threads for loading data")
        self.parser.add_argument("--seed", default=0, type=int, help="random seed")

        # audio arguments
        self.parser.add_argument("--audio_window", default=65535, type=int, help="audio segment length")
        self.parser.add_argument("--audio_sampling_rate", default=11025, type=int, help="sound sampling rate")
        self.parser.add_argument("--stft_frame", default=1022, type=int, help="stft frame length")
        self.parser.add_argument("--stft_hop", default=256, type=int, help="stft hop length")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.mode = self.mode

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # I should process the opt here, like gpu ids, etc.
        args = vars(self.opt)
        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        # save to the disk
        self.opt.name = (
            self.opt.name
            + "_graph_net_"
            + str(self.opt.gnet_type)
            + "_graph_lr_"
            + str(self.opt.lr_graph_net)
            + "_graph_wt_"
            + str(self.opt.graph_loss_weight)
            + "_batchSize_"
            + str(self.opt.batchSize)
        )
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("------------ Options -------------\n")
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s\n" % (str(k), str(v)))
            opt_file.write("-------------- End ----------------\n")
        return self.opt
