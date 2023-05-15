# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import os
import time

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_loader import CreateDataLoader
from models import criterion
from models.audioVisual_model_noaug_noclassif import AudioVisualModel
from models.models import ModelBuilder
from options.train_options import TrainOptions
from scipy.misc import imsave
from torch.autograd import Variable
from utils import utils, viz


def create_optimizer(nets, opt):
    (net_visual, net_unet, net_classifier, graph_nets, map_net, rnn, rnn_classifier) = nets
    param_groups = [
        {"params": net_unet.parameters(), "lr": opt.lr_unet},
        {"params": net_classifier.parameters(), "lr": opt.lr_classifier},
        {"params": graph_nets.parameters(), "lr": opt.lr_graph_net},
        {"params": map_net.parameters(), "lr": opt.lr_classifier},
        {"params": rnn.parameters(), "lr": opt.lr_classifier},
    ]

    if opt.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == "adam":
        return torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay_factor


def save_visualization(vis_rows, outputs, batch_data, save_dir, opt):  # Need to fix ordering here
    # fetch data and predictions
    mag_mix = batch_data["audio_mix_mags"]
    phase_mix = batch_data["audio_mix_phases"]

    pred_masks_ = outputs["pred_mask"]
    gt_masks_ = outputs["gt_mask"]
    mag_mix_ = outputs["audio_mix_mags"]
    weight_ = outputs["weight"]
    visual_object = outputs["visual_object"]
    gt_label = outputs["gt_label"]
    _, pred_label = torch.max(output["pred_label"], 1)

    # unwarp log scale
    B = mag_mix.size(0)
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame // 2 + 1, gt_masks_.size(3), warp=False)).to(
            opt.device
        )
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
        gt_masks_linear = F.grid_sample(gt_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
        gt_masks_linear = gt_masks_

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    pred_masks_ = pred_masks_.detach().cpu().numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    gt_masks_ = gt_masks_.detach().cpu().numpy()
    gt_masks_linear = gt_masks_linear.detach().cpu().numpy()
    visual_object = visual_object.detach().cpu().numpy()
    gt_label = gt_label.detach().cpu().numpy()
    pred_label = pred_label.detach().cpu().numpy()

    print("Visualization Log: " + str(save_dir))
    # loop over each example
    for j in range(min(B, opt.num_visualization_examples)):  # Iterate over the batch samples one at a time
        row_elements = []

        print("GT is: " + str(int(gt_label[j])) + " Pred: " + str(int(pred_label[j])))  # * video names
        prefix = str(j) + "-" + label_list[int(gt_label[j])] + "-" + label_list[int(pred_label[j])]
        utils.mkdirs(os.path.join(save_dir, prefix))

        # save mixture
        mix_wav = utils.istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=opt.stft_hop)
        mix_amp = utils.magnitude2heatmap(mag_mix_[j, 0])
        weight = utils.magnitude2heatmap(weight_[j, 0], log=False, scale=100.0)
        filename_mixwav = os.path.join(prefix, "mix.wav")
        filename_mixmag = os.path.join(prefix, "mix.jpg")
        filename_weight = os.path.join(prefix, "weight.jpg")
        imsave(os.path.join(save_dir, filename_mixmag), mix_amp[::-1, :, :])
        imsave(os.path.join(save_dir, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(save_dir, filename_mixwav), opt.audio_sampling_rate, mix_wav)
        print("Sample Outputs: " + str(os.path.join(save_dir, filename_mixmag)))
        row_elements += [{"text": prefix}, {"image": filename_mixmag, "audio": filename_mixwav}]

        # GT and predicted audio reconstruction
        gt_mag = mag_mix[j, 0] * gt_masks_linear[j, 0]
        gt_wav = utils.istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=opt.stft_hop)
        pred_mag = mag_mix[j, 0] * pred_masks_linear[j, 0]
        preds_wav = utils.istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=opt.stft_hop)

        # output masks
        filename_gtmask = os.path.join(prefix, "gtmask.jpg")
        filename_predmask = os.path.join(prefix, "predmask.jpg")
        gt_mask = (np.clip(gt_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        pred_mask = (np.clip(pred_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(save_dir, filename_gtmask), gt_mask[::-1, :])
        imsave(os.path.join(save_dir, filename_predmask), pred_mask[::-1, :])

        # ouput spectrogram (log of magnitude, show colormap)
        filename_gtmag = os.path.join(prefix, "gtamp.jpg")
        filename_predmag = os.path.join(prefix, "predamp.jpg")
        gt_mag = utils.magnitude2heatmap(gt_mag)
        pred_mag = utils.magnitude2heatmap(pred_mag)
        imsave(os.path.join(save_dir, filename_gtmag), gt_mag[::-1, :, :])
        imsave(os.path.join(save_dir, filename_predmag), pred_mag[::-1, :, :])

        # output audio
        filename_gtwav = os.path.join(prefix, "gt.wav")
        filename_predwav = os.path.join(prefix, "pred.wav")
        wavfile.write(os.path.join(save_dir, filename_gtwav), opt.audio_sampling_rate, gt_wav)
        wavfile.write(os.path.join(save_dir, filename_predwav), opt.audio_sampling_rate, preds_wav)
        print("Sample Audio Outputs: " + str(os.path.join(save_dir, filename_predwav)))

        row_elements += [
            {"image": filename_predmag, "audio": filename_predwav},
            {"image": filename_gtmag, "audio": filename_gtwav},
            {"image": filename_predmask},
            {"image": filename_gtmask},
        ]

        row_elements += [{"image": filename_weight}]
        vis_rows.append(row_elements)


# used to display validation loss
def display_val(model, crit, writer, index, dataset_val, rnn_init, opt):
    # remove previous viz results
    save_dir = os.path.join(".", opt.checkpoints_dir, opt.name, "visualization")
    utils.mkdirs(save_dir)

    # initial results lists
    classifier_losses = []
    coseparation_losses = []
    rnn_classifier_losses = []
    # initialize HTML header
    visualizer = viz.HTMLVisualizer(os.path.join(save_dir, "index.html"))
    header = ["Filename", "Input Mixed Audio"]
    header += ["Predicted Audio" "GroundTruth Audio", "Predicted Mask", "GroundTruth Mask", "Loss weighting"]
    visualizer.add_header(header)
    vis_rows = []

    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if True:
                output = model.forward(val_data, rnn_init=rnn_init)
                loss_ind = val_data["loss_ind"].cuda(opt.gpu_ids[0])
                loss_classification = crit["loss_classification"]
                classifier_loss = (
                    loss_classification(output["pred_label"], Variable(output["gt_label"], requires_grad=False))
                    * loss_ind
                ).mean() * opt.classifier_loss_weight
                coseparation_loss = get_coseparation_loss(output, opt, crit["loss_coseparation"], loss_ind)
                rnn_classif_loss = output["rnn_cls"]
                classifier_losses.append(classifier_loss.item())
                coseparation_losses.append(coseparation_loss.item())
                rnn_classifier_losses.append(rnn_classif_loss.item())

                if i == (opt.validation_batches - 1) and opt.validation_visualization:  # *else:
                    output = model.forward(val_data)
                    save_visualization(vis_rows, output, val_data, save_dir, opt)  # visualize one batch
                    break

    avg_classifier_loss = sum(classifier_losses) / len(classifier_losses)
    avg_coseparation_loss = sum(coseparation_losses) / len(coseparation_losses)
    avg_rnn_classifier_loss = sum(rnn_classifier_losses) / len(rnn_classifier_losses)

    if opt.tensorboard:
        writer.add_scalar("data/val_classifier_loss", avg_classifier_loss, index)
        writer.add_scalar("data/val_coseparation_loss", avg_coseparation_loss, index)
        writer.add_scalar("data/val_rnn_classifier_loss", avg_rnn_classifier_loss, index)

    print("val classifier loss: %.3f" % avg_classifier_loss)
    print("val coseparation loss: %.3f" % avg_coseparation_loss)
    print("val rnn classifier loss: %.3f" % avg_rnn_classifier_loss)
    return avg_coseparation_loss + avg_classifier_loss + avg_rnn_classifier_loss


def get_coseparation_loss(output, opt, loss_coseparation, loss_ind):
    # initialize a dic to store the index of the list
    vid_index_dic = {}
    vids = output["vids"].squeeze(1).cpu().numpy()
    O = vids.shape[0]
    count = 0
    for i in range(O):
        if not vids[i] in vid_index_dic:
            vid_index_dic[vids[i]] = count
            count = count + 1

    # initialize three lists of length = number of video clips to reconstruct
    predicted_mask_list = [None for i in range(len(vid_index_dic.keys()))]
    gt_mask_list = [None for i in range(len(vid_index_dic.keys()))]
    weight_list = [None for i in range(len(vid_index_dic.keys()))]

    # iterate through all objects
    gt_masks = output["gt_mask"]
    mask_prediction = output["pred_mask"]
    weight = output["weight"]
    for i in range(O):
        if predicted_mask_list[vid_index_dic[vids[i]]] is None:
            gt_mask_list[vid_index_dic[vids[i]]] = gt_masks[i, :, :, :]
            weight_list[vid_index_dic[vids[i]]] = weight[i, :, :, :]
            predicted_mask_list[vid_index_dic[vids[i]]] = (
                mask_prediction[i, :, :, :] * loss_ind[i]
            )  # Multiply with the indicator to enforce loss
        else:
            predicted_mask_list[vid_index_dic[vids[i]]] = predicted_mask_list[vid_index_dic[vids[i]]] + (
                mask_prediction[i, :, :, :] * loss_ind[i]
            )

    if opt.mask_loss_type == "BCE":
        for i in range(O):
            # clip the prediction results to make it in the range of [0,1] for BCE loss
            predicted_mask_list[vid_index_dic[vids[i]]] = torch.clamp(predicted_mask_list[vid_index_dic[vids[i]]], 0, 1)
    coseparation_loss = loss_coseparation(predicted_mask_list, gt_mask_list, weight_list)
    return coseparation_loss


# parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")


# Network Builders
builder = ModelBuilder()

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


@torch.no_grad()
def reorg_gt(pred_label, gt_label, loss_ind):
    loss_ind_reshape = loss_ind.view(-1, opt.num_object_per_video + 1)
    loss_ind_chk = torch.sum(loss_ind_reshape, dim=1) == (opt.num_object_per_video + 1)  # Check if it is a duet video

    loss_std = F.cross_entropy(pred_label, gt_label, reduction="none").view(-1, opt.num_object_per_video + 1)
    # Flip the gt_label
    if opt.num_object_per_video == 2:
        tmp = gt_label.view(-1, opt.num_object_per_video + 1)
        new_gt = torch.index_select(
            tmp, 1, torch.LongTensor([1, 0, 2]).cuda()
        )  # Flip the first and second object, third is background
        loss_flip = F.cross_entropy(pred_label, new_gt.view(-1), reduction="none").view(
            -1, opt.num_object_per_video + 1
        )

    gt_label = gt_label.view(-1, opt.num_object_per_video + 1)
    # Iterate over each video and if it is a duet, flip gt indices
    for i in range(new_gt.size(0)):
        if loss_ind_chk[i]:  # Check if it is a duet video
            if loss_flip[i].sum() < loss_std[i].sum():  # Check if the flipped labels are better
                gt_label[i, :] = new_gt[i, :]

    return gt_label.view(-1)


def graph_declare():
    return builder.build_graph_encoder(
        feat_dim=512,
        heads=4,
        hidden_act=opt.hidden_act,
        fin_graph_rep=opt.graph_enc_dim,
        pooling_ratio=opt.pooling_ratio,
        nos_classes=opt.number_of_classes,
        gnet_type=opt.gnet_type,
    )


graph_nets = graph_declare()  # Graph Embedding

map_net = nn.Sequential(
    nn.Linear(opt.feat_dim, opt.feat_dim // 2), nn.LeakyReLU(negative_slope=0.2), nn.Linear(opt.feat_dim // 2, 512)
)

# Define the recurrent layer
rnn = nn.GRU(input_size=512, hidden_size=512, num_layers=1)

# *# Define the RNN classifier
rnn_classif = None

nets = (None, net_unet, net_classifier, graph_nets, map_net, rnn, rnn_classif)  # * net_visual
# *print(nets)
# construct data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print("#training images = %d" % dataset_size)

# create validation set data loader if validation_on option is set
if opt.validation_on:
    # temperally set to val to load val data
    opt.mode = "val"
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print("#validation images = %d" % dataset_size_val)
    opt.mode = "train"  # set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)

# Set up optimizer
optimizer = create_optimizer(nets, opt)

# Set up loss functions
loss_classification = criterion.CELoss()
if opt.mask_loss_type == "L1":
    loss_coseparation = criterion.L1Loss()
elif opt.mask_loss_type == "L2":
    loss_coseparation = criterion.L2Loss()
elif opt.mask_loss_type == "BCE":
    loss_coseparation = criterion.BCELoss()
if len(opt.gpu_ids) > 0:
    loss_classification.cuda(opt.gpu_ids[0])
    loss_coseparation.cuda(opt.gpu_ids[0])
crit = {"loss_classification": loss_classification, "loss_coseparation": loss_coseparation}

# initialization
total_batches = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_graph_sep_loss = []
batch_classifier_loss = []
batch_coseparation_loss = []
batch_rnn_classif_loss = []
best_err = float("inf")
rnn_init = torch.zeros(rnn.num_layers, opt.num_per_mix * opt.batchSize, rnn.hidden_size).cuda(opt.gpu_ids[0])

for epoch in range(1 + opt.epoch_count, opt.niter + 1):
    torch.cuda.synchronize()
    epoch_start_time = time.time()

    if opt.measure_time:
        iter_start_time = time.time()
    for i, data in enumerate(dataset):
        if opt.measure_time:
            torch.cuda.synchronize()
            iter_data_loaded_time = time.time()

        total_batches += 1

        # forward pass
        model.zero_grad()
        output = model.forward(data, rnn_init=rnn_init, opt=opt)

        # Loss weight indicator
        loss_ind = data["loss_ind"].cuda(opt.gpu_ids[0])
        # compute loss
        # Compute the correct matching indices for foreground objects in Duet Videos, mostly
        gt_labels = reorg_gt(output["pred_label"], Variable(output["gt_label"], requires_grad=False), loss_ind)

        # classifier_loss
        classifier_loss = (
            loss_classification(output["pred_label"], gt_labels) * loss_ind
        ).mean() * opt.classifier_loss_weight

        # coseparation loss
        coseparation_loss = (
            get_coseparation_loss(output, opt, loss_coseparation, loss_ind) * opt.coseparation_loss_weight
        )

        #
        rnn_classif_loss = output["rnn_cls"]

        if opt.measure_time:
            torch.cuda.synchronize()
            iter_data_forwarded_time = time.time()

        # store losses for this batch
        batch_classifier_loss.append(classifier_loss.item())
        batch_coseparation_loss.append(coseparation_loss.item())
        batch_rnn_classif_loss.append(rnn_classif_loss.item())

        optimizer.zero_grad()
        rnn_classif_loss.backward(retain_graph=True)
        classifier_loss.backward(retain_graph=True)
        coseparation_loss.backward()
        optimizer.step()

        if opt.measure_time:
            torch.cuda.synchronize()
            iter_model_backwarded_time = time.time()

        if opt.measure_time:
            torch.cuda.synchronize()
            iter_model_backwarded_time = time.time()
            data_loading_time.append(iter_data_loaded_time - iter_start_time)
            model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
            model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

        if total_batches % opt.display_freq == 0:
            print("Display training progress at (epoch %d, total_batches %d)" % (epoch, total_batches))
            avg_classifier_loss = sum(batch_classifier_loss) / len(batch_classifier_loss)
            avg_coseparation_loss = sum(batch_coseparation_loss) / len(batch_coseparation_loss)
            avg_rnn_classif_loss = sum(batch_rnn_classif_loss) / len(batch_rnn_classif_loss)

            print(
                "classifier loss: %.3f, co-separation loss: %.3f, rnn_classifier loss: %.3f"
                % (avg_classifier_loss, avg_coseparation_loss, avg_rnn_classif_loss)
            )
            batch_classifier_loss = []
            batch_coseparation_loss = []
            batch_rnn_classif_loss = []
            if opt.tensorboard:
                writer.add_scalar("data/classifier_loss", avg_classifier_loss, i)
                writer.add_scalar("data/coseparation_loss", avg_coseparation_loss, i)

            if opt.measure_time:
                print("average data loading time: %.3f" % (sum(data_loading_time) / len(data_loading_time)))
                print("average forward time: %.3f" % (sum(model_forward_time) / len(model_forward_time)))
                print("average backward time: %.3f" % (sum(model_backward_time) / len(model_backward_time)))
                data_loading_time = []
                model_forward_time = []
                model_backward_time = []
            print("end of display \n")

        if total_batches % opt.save_latest_freq == 0:
            print("saving the latest model (epoch %d, total_batches %d)" % (epoch, total_batches))
            torch.save(
                net_unet.state_dict(),
                os.path.join(".", opt.checkpoints_dir, opt.name, "unet_" + str(total_batches) + ".pth"),
            )
            torch.save(
                net_classifier.state_dict(),
                os.path.join(".", opt.checkpoints_dir, opt.name, "classifier_" + str(total_batches) + ".pth"),
            )
            torch.save(
                graph_nets.state_dict(),
                os.path.join(".", opt.checkpoints_dir, opt.name, "graph_net_" + str(total_batches) + ".pth"),
            )
            torch.save(
                map_net.state_dict(),
                os.path.join(".", opt.checkpoints_dir, opt.name, "map_net_" + str(total_batches) + ".pth"),
            )
            torch.save(
                rnn.state_dict(), os.path.join(".", opt.checkpoints_dir, opt.name, "rnn_" + str(total_batches) + ".pth")
            )

        if total_batches % opt.validation_freq == 0 and opt.validation_on:
            model.eval()
            opt.mode = "val"
            print("Display validation results at (epoch %d, total_batches %d)" % (epoch, total_batches))
            val_err = display_val(model, crit, writer, total_batches, dataset_val, rnn_init, opt)
            print("end of display \n")
            model.train()
            opt.mode = "main"
            # save the model that achieves the smallest validation error
            if val_err < best_err:
                best_err = val_err

                print(
                    "saving the best model (epoch %d, total_batches %d) with validation error %.3f\n"
                    % (epoch, total_batches, val_err)
                )
                torch.save(net_unet.state_dict(), os.path.join(".", opt.checkpoints_dir, opt.name, "unet_best.pth"))
                torch.save(
                    net_classifier.state_dict(), os.path.join(".", opt.checkpoints_dir, opt.name, "classifier_best.pth")
                )
                torch.save(
                    graph_nets.state_dict(), os.path.join(".", opt.checkpoints_dir, opt.name, "graph_net_best.pth")
                )
                torch.save(map_net.state_dict(), os.path.join(".", opt.checkpoints_dir, opt.name, "map_net_best.pth"))
                torch.save(rnn.state_dict(), os.path.join(".", opt.checkpoints_dir, opt.name, "rnn_best.pth"))

        # decrease learning rate
        if total_batches in opt.lr_steps:
            decrease_learning_rate(optimizer, opt.decay_factor)
            print("decreased learning rate by ", opt.decay_factor)

        if opt.measure_time:
            torch.cuda.synchronize()
            iter_start_time = time.time()
