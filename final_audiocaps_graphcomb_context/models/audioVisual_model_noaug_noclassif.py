# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import warpgrid


class AudioVisualModel(torch.nn.Module):
    def name(self):
        return "AudioVisualModel"

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt

        # initialize model and criterions
        (
            self.net_visual,
            self.net_unet,
            self.net_classifier,
            self.graph_nets,
            self.map_net,
            self.rnn,
            self.rnn_classif,
        ) = nets

    def forward(self, input, rnn_init, mode="train", opt=None):
        labels = input["labels"].squeeze().long()

        vids = input["vids"]

        batch_vec = input["batch_vec"]
        audio_mags = input["audio_mags"]
        audio_mix_mags = input["audio_mix_mags"]

        audio_mix_mags = audio_mix_mags + 1e-10

        persons = input["per_visuals"]
        if mode != "test":  # If in training mode or val mode
            edge = input["edges"]

            if (
                edge.size(1) > 0
            ):  # If there is atleast one node in the batch for which graph processing needs to be done

                person_feature = self.map_net(
                    persons
                )  # * Nos. of (person) objects x 2048 -> Nos. of (person) objects x 512
                batch_vec = input["batch_vec"]  # *.to(self.opt.device) #*input['node_list_obj']
                # *print('batch_vec: ' + str(batch_vec))

                x = person_feature.view(
                    -1, person_feature.size(1)
                )  # Interleaved tensor with person and instrument nodes
                # *batch_vec = torch.cat([batch_vec, input['node_list_per']], dim=0)

                # Reshape the label tensor for RNN processing
                rnn_labels = labels.view(-1, self.opt.num_object_per_video + 1)
                # Forward through the graph embedding network
                if torch.min(batch_vec) <= 0:
                    graph_embed = self.graph_nets((x, edge, batch_vec))[1:, :]  # * Ignore the 0-person features
                else:
                    batch_vec = batch_vec - 1
                    graph_embed = self.graph_nets((x, edge, batch_vec))
                # Initialize the squared cosine similarity
                dot_p2 = 0
                # Run the recurrence through the RNN
                for t in range(self.opt.num_object_per_video + 1):
                    # Forward pass through the RNN
                    if t == 0:  # For the first object
                        g_e, rnn_init = self.rnn(graph_embed.unsqueeze(0), rnn_init)
                        g_e = g_e.squeeze()
                        # Normalize the RNN Output
                        cl = g_e / (g_e.norm(dim=1) + 1e-10)[:, None]
                        cl = cl.unsqueeze(1)
                    else:
                        tmp, rnn_init = self.rnn(graph_embed.unsqueeze(0), rnn_init)
                        tmp = tmp.squeeze()
                        # Augment it to the previous embeddings
                        g_e = torch.cat([g_e, tmp], dim=1)
                        # Normalize the RNN Output
                        tmp = (tmp / (tmp.norm(dim=1) + 1e-10)[:, None]).unsqueeze(1)
                        # Compute the squared cosine similarity
                        dot_p2 += (torch.bmm(cl, tmp.transpose(1, 2)).mean()) ** 2
                        # Augment the current RNN output to the ones from the previous time steps
                        cl = torch.cat([cl, tmp], dim=1)

                # Compute the feature for U-Net
                graph_embed = g_e.view(-1, person_feature.size(1))

                visual_feature = graph_embed.unsqueeze(2).unsqueeze(
                    2
                )  # Extract out only the attended instrument features #* [:, person_feature.size(1):]

            else:
                visual_feature = self.net_visual(Variable(persons, requires_grad=False))  # * self.map_net()

            # warp the spectrogram
            B = audio_mix_mags.size(0)
            T = audio_mix_mags.size(3)
            if self.opt.log_freq:
                grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
                audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
                audio_mags = F.grid_sample(audio_mags, grid_warp)

            # calculate ground-truth masks
            gt_masks = audio_mags / audio_mix_mags
            # clamp to avoid large numbers in ratio masks
            gt_masks.clamp_(0.0, 5.0)

            # audio-visual feature fusion through UNet and predict mask
            audio_log_mags = torch.log(
                audio_mix_mags
            ).detach()  # *; print('Shape of audio_log_mags: ' + str(audio_log_mags.size()));
            mask_prediction = self.net_unet(audio_log_mags, visual_feature)  # *) torch.randn(26, 512, 1, 1).cuda()

            # masking the spectrogram of mixed audio to perform separation
            separated_spectrogram = audio_mix_mags * mask_prediction

            # generate spectrogram for the classifier
            spectrogram2classify = torch.log(separated_spectrogram + 1e-10)  # get log spectrogram

            # calculate loss weighting coefficient
            if self.opt.weighted_loss:
                weight = torch.log1p(audio_mix_mags)
                weight = torch.clamp(weight, 1e-3, 10)
            else:
                weight = None

            # classify the predicted spectrogram
            label_prediction = self.net_classifier(spectrogram2classify)

            output = {
                "gt_label": labels,
                "pred_label": label_prediction,
                "pred_mask": mask_prediction,
                "gt_mask": gt_masks,
                "rnn_labels": rnn_labels.long(),
                "rnn_cls": dot_p2,
                "pred_spectrogram": separated_spectrogram,
                "visual_object": None,
                "audio_mix_mags": audio_mix_mags,
                "weight": weight,
                "vids": vids,
            }  # * labels.view(-1).long()
        else:  # If in test mode, assuming batch size is 1

            person_feature = self.map_net(persons)
            person_feature = person_feature.view(
                -1, person_feature.size(1)
            )  # * Nos. of (person) objects x 2048 -> Nos. of (person) objects x 512
            batch_vec = input["batch_vec"]  # *.to(self.opt.device) #*input['node_list_obj']
            edge = input["edges"]

            if edge is not None:  # If there is atleast one person detected in the batch
                graph_ip = person_feature

                visual_feature = self.graph_nets((graph_ip, edge, batch_vec))

                visual_feature = visual_feature.view(
                    1, -1
                )  # Assuming only 1 object, consider only the updated instrument feature

                # *print('Attention Weights are: ' + str(attn_wt))
            else:
                visual_feature = person_feature  # *self.map_net(x).view(1, -1)

            # Obtain the RNN output for this step
            g_e, _ = self.rnn(visual_feature.unsqueeze(0), rnn_init)
            visual_feature = g_e.squeeze(0)
            # *print('Size of visual_feature: ' + str(visual_feature.size()))
            # warp the spectrogram
            B = audio_mix_mags.size(0)
            T = audio_mix_mags.size(3)
            if self.opt.log_freq:
                grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
                audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
                audio_mags = F.grid_sample(audio_mags, grid_warp)

            # calculate ground-truth masks
            gt_masks = audio_mags / audio_mix_mags
            # clamp to avoid large numbers in ratio masks
            gt_masks.clamp_(0.0, 5.0)

            # audio-visual feature fusion through UNet and predict mask
            audio_log_mags = torch.log(
                audio_mix_mags
            ).detach()  # *; print('Shape of audio_log_mags: ' + str(audio_log_mags.size()));
            # *print('Size of audio_log_mags: ' + str(audio_log_mags.size()) + ' Size of visual_feature: ' + str(visual_feature.size()))
            mask_prediction = self.net_unet(
                audio_log_mags, visual_feature.unsqueeze(2).unsqueeze(2)
            )  # *) torch.randn(26, 512, 1, 1).cuda()

            # masking the spectrogram of mixed audio to perform separation
            separated_spectrogram = audio_mix_mags * mask_prediction

            # generate spectrogram for the classifier
            spectrogram2classify = torch.log(separated_spectrogram + 1e-10)  # get log spectrogram

            # calculate loss weighting coefficient
            if self.opt.weighted_loss:
                weight = torch.log1p(audio_mix_mags)
                weight = torch.clamp(weight, 1e-3, 10)
            else:
                weight = None

            # classify the predicted spectrogram
            label_prediction = self.net_classifier(spectrogram2classify)

            output = {
                "gt_label": labels,
                "pred_label": label_prediction,
                "pred_mask": mask_prediction,
                "pred_spectrogram": separated_spectrogram,
                "visual_object": None,
                "audio_mix_mags": audio_mix_mags,
                "weight": weight,
                "vids": vids,
            }

        return output
