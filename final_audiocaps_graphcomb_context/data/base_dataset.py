#!/usr/bin/env python3

# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0

import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return "BaseDataset"

    def initialize(self, opt):
        pass
