# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import os
import shutil

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._six import container_abcs, int_classes, string_classes

plt.switch_backend("Agg")
plt.ioff()


def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv + 1) / 2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def magnitude2heatmap(mag, log=True, scale=200.0):
    if log:
        mag = np.log10(mag + 1.0)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def mkdirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path)


def visualizeSpectrogram(spectrogram, save_path):
    fig, ax = plt.subplots(1, 1)
    plt.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.pcolormesh(librosa.amplitude_to_db(spectrogram))
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def istft_reconstruction(mag, phase, hop_length=256, length=65535):
    spec = mag.astype(np.complex) * np.exp(1j * phase)
    wav = librosa.istft(spec, hop_length=hop_length, length=length)
    return np.clip(wav, -1.0, 1.0)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
    nets (network list)   -- a list of networks
    requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# define customized collate to combine useful objects across video pairs
error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
numpy_type_map = {
    "float64": torch.DoubleTensor,
    "float32": torch.FloatTensor,
    "float16": torch.HalfTensor,
    "int64": torch.LongTensor,
    "int32": torch.IntTensor,
    "int16": torch.ShortTensor,
    "int8": torch.CharTensor,
    "uint8": torch.ByteTensor,
}


def object_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        tot_nodes, edge_ten, out = 0, [], None
        # Iterate over each edge-tensor in the batch
        if batch[0].size(0) == 2:  # For processing edge tensor
            for b in batch:
                edge_ten.append(
                    b + tot_nodes
                )  # Increment the edge tensor by cumul. nos. nodes across all prior samples in this batch
                tot_nodes += torch.max(b) + 1  # Since fully-connected so max gives us the total number of nodes
                out = torch.cat(edge_ten, 1)

            if out is None:  # No person detetced for any sample in the batch
                out = torch.zeros(2, 0).int()
        else:  # For processing loss indicator tensor
            out = torch.cat(batch, dim=0).view(-1)
        return out
    elif elem_type.__module__ == "numpy":
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            ten = []
            for b in batch:
                t = torch.from_numpy(b).float()
                ten.append(t)

            return torch.cat(ten, 0)  # concatenate even if dimension differs

        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith("float") else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], list):  # If the data is of type list, then it is batch_vec indicator
        batch_vec, tot_grp = [], 0
        for cntr, b in enumerate(batch):
            ind_arr = np.array(b)
            z_arr = np.zeros_like(ind_arr).astype(np.int32)
            tmp = np.where(ind_arr == 0, z_arr, (ind_arr + tot_grp))
            tot_grp += np.amax(ind_arr)
            batch_vec.extend(list(tmp))

        return torch.Tensor(batch_vec).long()  # Return the tensor

    elif isinstance(batch[0], float):
        tot_graph, b_ind, out = 0, [], None

        for b in batch:
            if b > 0:
                b_ind.append(torch.arange(int(b)).view(-1, 1).repeat(1, 2).view(-1) + tot_graph)  #
                tot_graph += int(b)

        if len(b_ind) > 0:
            return torch.cat(b_ind, dim=0).long()  # *.view(-1)
        else:
            # *print('b_ind size: ' + str(b_ind.size()))
            return torch.Tensor([])  # *, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: object_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [object_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))
