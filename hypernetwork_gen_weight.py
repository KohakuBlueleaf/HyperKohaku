#!/usr/bin/env python
# coding=utf-8

# Modified by KohakuBlueLeaf
# Modified from diffusers/example/dreambooth/train_dreambooth_lora.py
# see original licensed below
# =======================================================================
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# =======================================================================

import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, Resize, ToTensor, Compose, Normalize
from PIL import Image
from PIL.ImageOps import exif_transpose

import diffusers
from diffusers.loaders import LoraLoaderMixin
from diffusers import (
    UNet2DConditionModel,
)

from modules.lightlora import LiLoRAAttnProcessor, LiLoRAXformersAttnProcessor
from modules.hypernet import HyperDream


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--hyperkohaku_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained hyperkohaku model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=None,
        required=True,
        help="Path to reference image",
    )
    parser.add_argument(
        "--decode_iter",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--down_dim",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--up_dim",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def unet_lilora_attn_processors_state_dict(unet):
    r"""
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def main(args):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    unet_lora_attn_procs = {}
    unet_lora_linear_layers = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        module = LiLoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, 
            rank=args.rank, down_dim=args.down_dim, up_dim=args.up_dim,
        )
        unet_lora_linear_layers.extend(module.layers)
        unet_lora_attn_procs[name] = module
    unet.set_attn_processor(unet_lora_attn_procs)
    hypernetwork = HyperDream(
        weight_dim=(args.down_dim + args.up_dim)*args.rank,
        weight_num=len(unet_lora_linear_layers),
        sample_iters=args.decode_iter,
    )
    hypernetwork.set_lilora(unet_lora_linear_layers)
    
    if os.path.isdir(args.hyperkohaku_model_path):
        path = os.path.join(args.hyperkohaku_model_path, "hypernetwork.bin")
        hypernetwork.load_state_dict(torch.load(path)['hypernetwork'])
    else:
        hypernetwork.load_state_dict(torch.load(args.hyperkohaku_model_path['hypernetwork']))
    
    hypernetwork = hypernetwork.to("cuda")
    img_transpose = Compose([
        Resize(512),
        CenterCrop(512),
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])
    ref_img = Image.open(args.reference_image_path).convert("RGB")
    ref_img = img_transpose(ref_img).unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        hypernetwork(ref_img)
    
    for lilora in unet_lora_linear_layers:
        lilora.make_weight()
    
    attn_processors_state_dict = unet_lilora_attn_processors_state_dict(unet)
    LoraLoaderMixin.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=attn_processors_state_dict,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)