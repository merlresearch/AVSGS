#!/usr/bin/env python3

# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2019 Ruohan Gao
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: CC-BY-4.0


def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader

    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
