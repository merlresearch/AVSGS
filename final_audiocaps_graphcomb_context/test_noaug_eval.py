#!/usr/bin/env python
# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import csv
import json
import os

import h5py
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.audioVisual_dataset import compute_overlap, generate_spectrogram_magphase
from mir_eval.separation import bss_eval_sources
from models.audioVisual_model_noaug_noclassif import AudioVisualModel
from models.models import ModelBuilder
from options.test_options import TestOptions
from utils import utils


def clip_audio(audio):
    audio[audio > 1.0] = 1.0
    audio[audio < -1.0] = -1.0
    return audio


def get_separated_audio(outputs, batch_data, opt):
    # fetch data and predictions
    mag_mix = batch_data["audio_mix_mags"]
    phase_mix = batch_data["audio_mix_phases"]
    pred_masks_ = outputs["pred_mask"]
    mag_mix_ = outputs["audio_mix_mags"]
    # unwarp log scale
    B = mag_mix.size(0)
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame // 2 + 1, pred_masks_.size(3), warp=False)).to(
            opt.device
        )
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    pred_mag = mag_mix[0, 0] * pred_masks_linear[0, 0]
    preds_wav = utils.istft_reconstruction(pred_mag, phase_mix[0, 0], hop_length=opt.stft_hop, length=opt.audio_window)
    return preds_wav


def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
    # print reference_sources.shape
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
    # print estimated_sources.shape
    (sdr, sir, sar, perm) = bss_eval_sources(np.asarray(reference_sources), np.asarray(estimated_sources), False)
    # print sdr, sir, sar, perm
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def graph_declare(builder, opt, model_nm):
    return builder.build_graph_encoder(
        feat_dim=512,
        heads=4,
        hidden_act=opt.hidden_act,
        fin_graph_rep=opt.graph_enc_dim,
        pooling_ratio=opt.pooling_ratio,
        nos_classes=opt.number_of_classes,
        gnet_type=opt.gnet_type,
        weights=model_nm,
    )


def main():
    # load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")

    # Network Builders
    builder = ModelBuilder()
    net_visual = None

    net_unet = builder.build_unet(
        unet_num_layers=opt.unet_num_layers,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_unet,
    )

    if opt.with_additional_scene_image:
        opt.number_of_classes = opt.number_of_classes + 1

    net_classifier = builder.build_classifier(
        pool_type=opt.classifier_pool,
        num_of_classes=opt.number_of_classes,
        input_channel=opt.unet_output_nc,
        weights=opt.weights_classifier,
    )

    graph_nets = graph_declare(builder, opt, opt.weights_graph)  # Define the graph encoder

    map_net = nn.Sequential(
        nn.Linear(opt.feat_dim, opt.feat_dim // 2), nn.LeakyReLU(negative_slope=0.2), nn.Linear(opt.feat_dim // 2, 512)
    )  # Skip BatchNorm to avoid averaging over 0-feature nodes. # Define the mapping network for person nodes

    map_net.load_state_dict(torch.load(opt.weights_map_net))  # Load the weights from the saved model

    # Define the recurrent layer
    rnn = nn.GRU(input_size=512, hidden_size=512, num_layers=1)
    rnn.load_state_dict(torch.load(opt.weights_rnn))  # Load the weights from the saved model

    # Define the RNN classifier
    rnn_classif = None

    nets = (net_visual, net_unet, net_classifier, graph_nets, map_net, rnn, rnn_classif)

    # construct our audio-visual model
    model = AudioVisualModel(nets, opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)
    model.eval()

    rnn_init = torch.zeros(rnn.num_layers, 1, rnn.hidden_size).cuda(opt.gpu_ids[0])  # Assume batchSize of 1 here
    print("All models loaded!")

    # load the two audios
    audio1_path = os.path.join(opt.data_path, "Extracted_Audio/test/", opt.video1_name + ".aac")
    audio1, _ = librosa.load(audio1_path, sr=opt.audio_sampling_rate)
    audio2_path = os.path.join(opt.data_path, "Extracted_Audio/test/", opt.video2_name + ".aac")
    audio2, _ = librosa.load(audio2_path, sr=opt.audio_sampling_rate)

    # make sure the two audios are of the same length and then mix them
    audio_length = min(len(audio1), len(audio2))
    audio1 = clip_audio(audio1[:audio_length])
    audio2 = clip_audio(audio2[:audio_length])
    audio_mix = (audio1 + audio2) / 2.0

    # Load the json file for the first video
    with open(
        os.path.join(opt.data_path, "Extracted_Frames/test/", opt.video1_name, "Image_info_20obj.json"), "r"
    ) as f:
        vid_meta_1 = json.load(f)

    # Open the h5py file for man/woman node for the first video
    obj_det_1 = h5py.File(
        os.path.join(opt.data_path, "Extracted_Frames/test/", opt.video1_name, "Image_feature_20obj.h5"), "r"
    )

    # Load the json file for the second video
    with open(
        os.path.join(opt.data_path, "Extracted_Frames/test/", opt.video2_name, "Image_info_20obj.json"), "r"
    ) as f:
        vid_meta_2 = json.load(f)

    # Open the h5py file for man/woman node for the second video
    obj_det_2 = h5py.File(
        os.path.join(opt.data_path, "Extracted_Frames/test/", opt.video2_name, "Image_feature_20obj.h5"), "r"
    )

    avged_sep_audio1 = np.zeros((audio_length))
    avged_sep_audio2 = np.zeros((audio_length))

    # Initialize the bbox file
    det_bbox1, det_bbox2 = [], []
    # Load the detection file
    with open(
        os.path.join(opt.data_path, "Extracted_Frames/test/", opt.video1_name, "Vision_Text_Labels.csv"), "r"
    ) as csv_file:
        csv_reader = csv.reader(csv_file)

        for r in csv_reader:
            det_bbox1.append(r)
            break  # Since just one object, so one bounding box should be enough

    # Load the detection file
    with open(
        os.path.join(opt.data_path, "Extracted_Frames/test/", opt.video2_name, "Vision_Text_Labels.csv"), "r"
    ) as csv_file:
        csv_reader = csv.reader(csv_file)

        for r in csv_reader:
            det_bbox2.append(r)
            break  # Since just one object, so one bounding box should be enough

    for i in range(opt.num_of_object_detections_to_use):

        # Get the person info for the first video
        im_id = int(det_bbox1[0][2])

        person_idx = [
            idx
            for idx, obj, obj_conf in zip(
                range(len(vid_meta_1[im_id]["objects"])),
                vid_meta_1[im_id]["objects"],
                vid_meta_1[im_id]["objects_conf"],
            )
            if (obj_conf > opt.conf_thresh) and (obj != int(det_bbox1[0][1]))
        ]

        # Compute IoUs for each principal object - context candidate
        person_visuals_1, match = torch.from_numpy(np.zeros((obj_det_1["features"].shape[2]))).view(1, -1).float(), 0
        if len(person_idx) > 0:  # * continue  Skip if no person was detected
            ovrlp_iou = [
                compute_overlap(obj_det_1["boxes"][im_id, int(det_bbox1[0][3]), :], obj_det_1["boxes"][im_id, p, :])
                for p in person_idx
            ]
            match, match_idx = max(ovrlp_iou), np.argmax(np.array(ovrlp_iou))  # Closest match

        person_visuals_1 = torch.from_numpy(obj_det_1["features"][im_id, int(det_bbox1[0][3]), :]).view(
            1, -1
        )  # Obtain the object specific info
        if match > opt.iou_thresh:
            # If atleast one context node is available, then include all valid context nodes
            for ovr, p in zip(ovrlp_iou, person_idx):
                if ovr > opt.iou_thresh:
                    # Current box meets confidence and overlap criteria
                    person_visuals_1 = torch.cat(
                        [person_visuals_1, torch.from_numpy(obj_det_1["features"][im_id, p, :]).view(1, -1)], dim=0
                    )  # Obtain the context-specific features

        # Get the person info for the second video
        im_id = int(det_bbox2[0][2])
        person_idx = [
            idx
            for idx, obj, obj_conf in zip(
                range(len(vid_meta_2[im_id]["objects"])),
                vid_meta_2[im_id]["objects"],
                vid_meta_2[im_id]["objects_conf"],
            )
            if (obj_conf > opt.conf_thresh) and (obj != int(det_bbox2[0][1]))
        ]  # Man/Woman/Young Man class

        # Compute IoUs for each principal object - context candidate
        person_visuals_2, match = torch.from_numpy(np.zeros((obj_det_2["features"].shape[2]))).view(1, -1).float(), 0
        if len(person_idx) > 0:  # Skip if no person was detected
            ovrlp_iou = [
                compute_overlap(obj_det_2["boxes"][im_id, int(det_bbox2[0][3]), :], obj_det_2["boxes"][im_id, p, :])
                for p in person_idx
            ]
            match, match_idx = max(ovrlp_iou), np.argmax(np.array(ovrlp_iou))  # Closest match

        person_visuals_2 = torch.from_numpy(obj_det_2["features"][im_id, int(det_bbox2[0][3]), :]).view(
            1, -1
        )  # Obtain the object specific info
        if match > opt.iou_thresh:
            # If atleast one context node is available, then include all valid context nodes
            for ovr, p in zip(ovrlp_iou, person_idx):
                if ovr > opt.iou_thresh:
                    # Current box meets confidence and overlap criteria
                    person_visuals_2 = torch.cat(
                        [person_visuals_2, torch.from_numpy(obj_det_2["features"][im_id, p, :]).view(1, -1)], dim=0
                    )  # Obtain the context-specific features

        # Check if person_visuals_1 is not only object, then construct edge tensor
        if person_visuals_1.size(0) == 1:
            edges_1 = torch.Tensor([[0], [0]]).long()
            batch_vec_1 = torch.Tensor([0]).long()
        else:
            sub, obj, bv = [], [], []
            for nd1 in range(person_visuals_1.size(0)):
                for nd2 in range(person_visuals_1.size(0)):
                    sub.append(nd1)
                    obj.append(nd2)
                # Incorporate batch_vec info
                bv.append(0)
            # Convert the lists to tensors
            edges_1 = torch.Tensor([sub, obj]).long()
            batch_vec_1 = torch.Tensor(bv).long()

        # Check if person_visuals_2 is not only object, then construct edge tensor
        if person_visuals_2.size(0) == 1:
            edges_2 = torch.Tensor([[0], [0]]).long()
            batch_vec_2 = torch.Tensor([0]).long()
        else:
            sub, obj, bv = [], [], []
            for nd1 in range(person_visuals_2.size(0)):
                for nd2 in range(person_visuals_2.size(0)):
                    sub.append(nd1)
                    obj.append(nd2)
                # Incorporate batch_vec info
                bv.append(0)
            # Convert the lists to tensors
            edges_2 = torch.Tensor([sub, obj]).long()
            batch_vec_2 = torch.Tensor(bv).long()

        # perform separation over the whole audio using a sliding window approach
        overlap_count = np.zeros((audio_length))
        sep_audio1 = np.zeros((audio_length))
        sep_audio2 = np.zeros((audio_length))
        sliding_window_start = 0
        data = {}
        samples_per_window = opt.audio_window
        while sliding_window_start + samples_per_window < audio_length:
            sliding_window_end = sliding_window_start + samples_per_window
            audio_segment = audio_mix[sliding_window_start:sliding_window_end]
            audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(
                audio_segment, opt.stft_frame, opt.stft_hop
            )
            data["audio_mix_mags"] = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
            data["audio_mix_phases"] = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
            data["real_audio_mags"] = data["audio_mix_mags"]  # dont' care for testing
            data["audio_mags"] = data["audio_mix_mags"]  # dont' care for testing
            # separate for video 1
            data["per_visuals"] = person_visuals_1
            data["edges"] = edges_1
            data["batch_vec"] = batch_vec_1
            data["labels"] = torch.FloatTensor(np.ones((1, 1)))  # don't care for testing
            data["vids"] = torch.FloatTensor(np.ones((1, 1)))  # don't care for testing
            outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode)
            reconstructed_signal = get_separated_audio(outputs, data, opt)
            sep_audio1[sliding_window_start:sliding_window_end] = (
                sep_audio1[sliding_window_start:sliding_window_end] + reconstructed_signal
            )
            # separate for video 2
            data["per_visuals"] = person_visuals_2
            data["edges"] = edges_2
            data["batch_vec"] = batch_vec_2
            data["labels"] = torch.FloatTensor(np.ones((1, 1)))  # don't care for testing
            data["vids"] = torch.FloatTensor(np.ones((1, 1)))  # don't care for testing
            outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode)
            reconstructed_signal = get_separated_audio(outputs, data, opt)
            sep_audio2[sliding_window_start:sliding_window_end] = (
                sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal
            )
            # update overlap count
            overlap_count[sliding_window_start:sliding_window_end] = (
                overlap_count[sliding_window_start:sliding_window_end] + 1
            )
            sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

        # deal with the last segment
        audio_segment = audio_mix[-samples_per_window:]
        audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop)
        data["audio_mix_mags"] = torch.FloatTensor(audio_mix_mags).unsqueeze(0)
        data["audio_mix_phases"] = torch.FloatTensor(audio_mix_phases).unsqueeze(0)
        data["real_audio_mags"] = data["audio_mix_mags"]  # dont' care for testing
        data["audio_mags"] = data["audio_mix_mags"]  # dont' care for testing
        # separate for video 1
        data["per_visuals"] = person_visuals_1
        data["edges"] = edges_1
        data["batch_vec"] = batch_vec_1
        data["labels"] = torch.FloatTensor(np.ones((1, 1)))  # don't care for testing
        data["vids"] = torch.FloatTensor(np.ones((1, 1)))  # don't care for testing
        outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode)
        reconstructed_signal = get_separated_audio(outputs, data, opt)
        sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal
        # separate for video 2
        data["per_visuals"] = person_visuals_2
        data["edges"] = edges_2
        data["batch_vec"] = batch_vec_2
        data["labels"] = torch.FloatTensor(
            np.ones((1, 1))
        )  # don't care for testing torch.Tensor([det_box2[1]]).long() #*
        data["vids"] = torch.FloatTensor(np.ones((1, 1)))  # don't care for testing
        outputs = model.forward(data, rnn_init=rnn_init, mode=opt.mode)
        reconstructed_signal = get_separated_audio(outputs, data, opt)
        sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal
        # update overlap count
        overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

        # divide the aggregated predicted audio by the overlap count
        avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count) * 2)
        avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count) * 2)

    separation1 = avged_sep_audio1 / opt.num_of_object_detections_to_use
    separation2 = avged_sep_audio2 / opt.num_of_object_detections_to_use

    # Close the detection files
    obj_det_1.close()
    obj_det_2.close()

    # Get the evaluation metrics scores
    sdr, sir, sar = getSeparationMetrics(separation1, separation2, audio1, audio2)
    # Display the performance
    print("sdr: %3f, sir: %3f, sar: %3f" % (sdr, sir, sar))


if __name__ == "__main__":
    main()
