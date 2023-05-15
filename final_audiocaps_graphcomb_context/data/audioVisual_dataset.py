#!/usr/bin/env python3

# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import csv
import json
import os.path
import pickle
import random
from random import randrange

import h5py
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image, ImageEnhance


def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag


def augment_audio(audio):
    audio = audio * (random.random() + 0.5)  # 0.5 - 1.5
    audio[audio > 1.0] = 1.0
    audio[audio < -1.0] = -1.0
    return audio


def sample_audio(audio, window):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start : (audio_start + window)]
    return audio_sample


def augment_image(image):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    return image


def get_vid_name(npy_path):
    # first 11 chars are the video id
    return os.path.basename(npy_path)[:11]


def get_clip_name(npy_path):
    return os.path.basename(npy_path)[:-4]


def get_frame_root(npy_path):
    frame_path = "/".join(npy_path.decode("UTF-8").split("/")[:-3])
    return frame_path.encode("UTF-8")  # * os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'frame')


def get_audio_root(npy_path):
    audio_root = "../MUSIC_Dataset/Extracted_Audio/"
    audio_path = "/".join(npy_path.decode("UTF-8").split("/")[-5:-3])
    return os.path.join(audio_root, audio_path).encode(
        "UTF-8"
    )  # * os.path.join(os.path.dirname(os.path.dirname(npy_path)), 'audio_11025')


def sample_object_detections(detection_bbs):
    class_index_clusters = {}  # get the indexes of the detections for each class
    for i in range(detection_bbs.shape[0]):
        if int(detection_bbs[i, 0]) in class_index_clusters.keys():
            class_index_clusters[int(detection_bbs[i, 0])].append(i)
        else:
            class_index_clusters[int(detection_bbs[i, 0])] = [i]
    detection2return, idx_lst = np.array([]), []
    for cls in class_index_clusters.keys():
        sampledIndex = random.choice(class_index_clusters[cls])
        idx_lst.append(detection_bbs[sampledIndex, 2:])  # Store the index of the bounding box
        if detection2return.shape[0] == 0:
            detection2return = np.expand_dims(detection_bbs[sampledIndex, :], axis=0)
        else:
            detection2return = np.concatenate(
                (detection2return, np.expand_dims(detection_bbs[sampledIndex, :], axis=0)), axis=0
            )  # Constructing a 2d array
    return detection2return, idx_lst


def compute_overlap(bbox_inst, bbox_plr):
    """Computes the overlap between instrument and player bounding boxes and returns the IoU of overlapping pixels"""
    x1min, y1min, x1max, y1max = bbox_inst
    x2min, y2min, x2max, y2max = bbox_plr
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)
    iw = max(xmax - xmin, 0)
    ih = max(ymax - ymin, 0)
    inters = iw * ih
    union = (x1max - x1min) * (y1max - y1min) + (x2max - x2min) * (y2max - y2min) - inters
    return inters * 1.0 / union


class ASIW(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.NUM_PER_MIX = opt.num_per_mix
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.audio_window = opt.audio_window
        random.seed(opt.seed)
        self.iou_thresh = 0.1
        self.conf_thresh = 0.4
        self.feature_dim = 2048
        self.num_object_per_video = opt.num_object_per_video  # Maximum number of objects present in a video
        # initialization
        self.detection_dic = {}  # gather the clips for each video
        # load the list of valid videos
        file_list_path = os.path.join(opt.hdf5_path, "Valid_Videos_Vis_Text.pickle")
        # Read in the file names
        with open(file_list_path, "rb") as f:
            train_vid_lst, val_vid_lst, test_vid_lst = pickle.load(f)
        # Assign the file names
        if opt.mode == "train":
            detections = train_vid_lst
            self.mode = opt.mode
        else:
            detections = test_vid_lst
            self.mode = "test"
        # Initialize the root
        self.data_path = opt.data_path
        # *h5f = h5py.File(h5f_path, 'r')
        # *detections = h5f['detection'][:]
        for detection in detections:
            vidname = detection  # get video id
            if vidname in self.detection_dic.keys():
                self.detection_dic[vidname].append(
                    os.path.join(self.data_path, "Extracted_Frames", self.mode, vidname, "Vision_Text_Labels.csv")
                )
            else:
                self.detection_dic[vidname] = [
                    os.path.join(self.data_path, "Extracted_Frames", self.mode, vidname, "Vision_Text_Labels.csv")
                ]

        if opt.mode != "train":
            vision_transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
            self.videos2Mix = [
                random.sample(self.detection_dic.keys(), self.NUM_PER_MIX)
                for _ in range(self.opt.batchSize * self.opt.validation_batches)
            ]
        elif opt.preserve_ratio:
            vision_transform_list = [transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()]
        else:
            vision_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
        if opt.subtract_mean:
            vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.vision_transform = transforms.Compose(vision_transform_list)

        # load hdf5 file of scene images
        if opt.with_additional_scene_image:
            h5f_path = os.path.join(opt.scene_path)
            h5f = h5py.File(h5f_path, "r")
            self.scene_images = h5f["image"][:]

    def __getitem__(self, index):
        if self.mode == "val":  # In order to validate on the same samples everytime
            videos2Mix = self.videos2Mix[index]
        else:
            videos2Mix = random.sample(self.detection_dic.keys(), self.NUM_PER_MIX)  # get videos to mix
        clip_det_paths = [None for n in range(self.NUM_PER_MIX)]
        clip_det_bbs = [None for n in range(self.NUM_PER_MIX)]
        idx_lst = [None for n in range(self.NUM_PER_MIX)]
        for n in range(self.NUM_PER_MIX):
            if self.mode == "val":
                clip_det_paths[n] = self.detection_dic[videos2Mix[n]]  # Validate on the first clip of the video always
            else:
                clip_det_paths[n] = random.choice(
                    self.detection_dic[videos2Mix[n]]
                )  # randomly sample a clip from *_1 to *_final

            # * Load the detection file
            with open(clip_det_paths[n], "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                # Initialize the bbox file
                det_bbox = []
                for r in csv_reader:
                    det_bbox.append(r)
            # Sample the relevant object bounding boxes
            clip_det_bbs[n], idx_lst[n] = sample_object_detections(
                np.array(det_bbox)
            )  # load the bbs for the clip and sample one from each class

        audios = [None for n in range(self.NUM_PER_MIX)]  # audios of mixed videos
        per_ind = [[] for n in range(self.NUM_PER_MIX)]  # audios of mixed videos
        ind_v, grph_cnt = [], 0
        person_visuals = []
        objects_labels = []
        objects_audio_mag = []
        objects_audio_phase = []
        objects_vids = []
        wt_ind = []
        objects_audio_mix_mag = []
        objects_audio_mix_phase = []
        tot_nodes, sub, obj = 0, [], []

        for n in range(self.NUM_PER_MIX):  # Iterate over each video to mix
            val_nodes, new_person_visuals, new_ind_v, new_sub, new_obj = [], [], [], [], []
            vid = random.randint(1, 100000000000)  # generate a unique video id
            audio_path = os.path.join(
                self.data_path, "Extracted_Audio", self.mode, clip_det_paths[n].split("/")[-2] + ".aac"
            )
            audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)  # * [2:-1]
            audio_segment = sample_audio(audio, self.audio_window)
            if self.opt.enable_data_augmentation and self.mode == "train":
                audio_segment = augment_audio(audio_segment)
            audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)
            detection_bbs = clip_det_bbs[n]
            audios[n] = audio_segment  # make a copy of the audio to mix later
            # Load the json file
            with open(os.path.join("/".join(clip_det_paths[n].split("/")[:-1]), "Image_info_20obj.json"), "r") as f:
                vid_meta = json.load(f)
            # Open the h5py file for man/woman node
            obj_det = h5py.File(
                os.path.join("/".join(clip_det_paths[n].split("/")[:-1]), "Image_feature_20obj.h5"), "r"
            )

            # Only keep num_object_per_video sound sources
            for i in range(min(detection_bbs.shape[0], self.num_object_per_video)):  # Iterate over each source detected
                label = int(detection_bbs[i, 0]) + 1  # Since label 0 is reserved for backgorund/extra label
                im_id = int(detection_bbs[i, 2])  # Get the im_id for the matched frame
                person_idx = [
                    idx
                    for idx, obj, obj_conf in zip(
                        range(len(vid_meta[im_id]["objects"])),
                        vid_meta[im_id]["objects"],
                        vid_meta[im_id]["objects_conf"],
                    )
                    if (obj_conf > self.conf_thresh) and (obj != int(detection_bbs[i, 1]))
                ]  # Context objects

                # Compute IoUs for each person candidate
                match = 0
                if len(person_idx) > 0:  # * continue  Skip if no person was detected
                    ovrlp_iou = [
                        compute_overlap(
                            obj_det["boxes"][im_id, int(detection_bbs[i, 3]), :], obj_det["boxes"][im_id, p, :]
                        )
                        for p in person_idx
                    ]
                    match = max(ovrlp_iou)  # Closest match

                if match > self.iou_thresh:
                    # If atleast one context node is available, then include all valid context nodes
                    for ovr, p in zip(ovrlp_iou, person_idx):
                        if ovr > self.iou_thresh:
                            new_person_visuals.append(
                                torch.from_numpy(obj_det["features"][im_id, p, :])
                            )  # Obtain the context node features
                            new_ind_v.extend([grph_cnt + 1])
                            val_nodes.extend([tot_nodes])
                            tot_nodes += 1
                    # Get the feature of the object
                    new_person_visuals.append(
                        torch.from_numpy(obj_det["features"][im_id, int(detection_bbs[i, 3]), :])
                    )  # Obtain the sound node features
                    objects_labels.append(label)
                    # make a copy of the audio spec for each object
                    objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                    objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                    objects_vids.append(vid)
                    per_ind[n].append(1)  # Indicator if context node exists
                    new_ind_v.extend([grph_cnt + 1])
                    # Incorporate edge information
                    val_nodes.extend([tot_nodes])
                    # Increment tot_nodes
                    tot_nodes += 1  # Always increment by 2, since there are 2 nodes a context node and an object node
                else:
                    # Zero context node in this case - only incorporate the sounding object feature
                    # Incorporate object information
                    new_person_visuals.append(
                        torch.from_numpy(obj_det["features"][im_id, int(detection_bbs[i, 3]), :])
                    )  # Obtain the sound node features
                    objects_labels.append(label)
                    # make a copy of the audio spec for each object
                    objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                    objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                    objects_vids.append(vid)
                    per_ind[n].append(0)  # Indicator if context node exists
                    new_ind_v.extend([grph_cnt + 1])  # Context node should not be part of the graph

                    # Incorporate edge information - only self loop here
                    val_nodes.extend([tot_nodes])
                    tot_nodes += 1  # Always increment by 1, since there is only an object node

                wt_ind.append(1)  # An indicator indicating if loss for this sample will be computed

            # Check if maximum nos. of objects covered, if not introduce audio input/gt and visual feature nodes for the difference
            if detection_bbs.shape[0] < self.num_object_per_video:
                nos_missing_obj = self.num_object_per_video - detection_bbs.shape[0]  # Store the difference count
                # Add audio signal for the missing elements
                for _ in range(nos_missing_obj):
                    new_person_visuals.append(torch.zeros(1, self.opt.feat_dim).float())  # Zero sound node in this case
                    objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                    objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                    objects_vids.append(vid)
                    objects_labels.append(0)
                    new_ind_v.extend([0])  # Context node should not be part of the graph
                    # Increment tot_nodes
                    tot_nodes += 1  # Always increment by 2, since there are 2 nodes both (zero/zero) person node and an object node
                    wt_ind.append(0)  # An indicator indicating if loss for this sample will be computed

            # add an additional scene image for each video
            if self.opt.with_additional_scene_image and (vid in objects_vids):
                new_person_visuals.append(
                    torch.from_numpy(obj_det["features"][0, 0, :])
                )  # Choose the first box of the first im_id as the random object
                objects_labels.append(0)
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)
                new_ind_v.extend([grph_cnt + 1])  # First element to be excluded eventually from graph
                # Incorporate edge information - only self loop here
                val_nodes.extend([tot_nodes])
                wt_ind.append(1)
                # Increment tot_nodes
                tot_nodes += 1  # Always increment by 1, since there is only an object node

            grph_cnt += 1
            obj_det.close()

            # Define the edges for this graph
            for v1 in val_nodes:  # Bidirectional, each graph has 2 nodes -- object + person
                for v2 in val_nodes:
                    new_sub.append(v1)
                    new_obj.append(v2)

            # Augment the batch_vec indicator
            ind_v.extend(new_ind_v)

            person_visuals.extend(new_person_visuals)
            sub.extend(new_sub)
            obj.extend(new_obj)

        # mix audio and make a copy of mixed audio spec for each object
        audio_mix = np.asarray(audios).sum(axis=0) / self.NUM_PER_MIX
        audio_mix_mag, audio_mix_phase = generate_spectrogram_magphase(audio_mix, self.stft_frame, self.stft_hop)
        for n in range(self.NUM_PER_MIX):
            detection_bbs = clip_det_bbs[n]
            for i in range(self.num_object_per_video):  # * zip(per_ind[n], range(detection_bbs.shape[0]))
                # *if ind == 1: # If person exists
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))

            if self.opt.with_additional_scene_image:
                objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
                objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))

        audio_mags = np.vstack(objects_audio_mag)  # audio spectrogram magnitude

        audio_phases = np.vstack(objects_audio_phase)  # audio spectrogram phase

        labels = np.vstack(objects_labels)  # labels for each object, 0 denotes padded object

        vids = np.vstack(objects_vids)  # video indexes for each object, each video should have a unique id

        audio_mix_mags = np.vstack(objects_audio_mix_mag)

        audio_mix_phases = np.vstack(objects_audio_mix_phase)

        edges = torch.Tensor([sub, obj]).long()

        data = {"labels": labels, "audio_mags": audio_mags, "audio_mix_mags": audio_mix_mags, "vids": vids}

        data["edges"] = edges

        data["batch_vec"] = ind_v  # Of List type
        data["loss_ind"] = torch.Tensor(wt_ind)

        if self.opt.mode == "val" or self.opt.mode == "test":
            data["audio_phases"] = audio_phases
            # *data['audio_phases_z'] = audio_phases_z
            data["audio_mix_phases"] = audio_mix_phases
            # *data['audio_mix_phases_z'] = audio_mix_phases_z

        data["per_visuals"] = np.vstack(person_visuals)

        return data

    def __len__(self):
        if self.opt.mode == "train":
            return self.opt.batchSize * self.opt.num_batch
        elif self.opt.mode == "val":
            return self.opt.batchSize * self.opt.validation_batches

    def name(self):
        return "ASIW"
