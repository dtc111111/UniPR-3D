# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class AggregatorFt(Aggregator):

    def __init__(self, only_return_final=True, **kwargs):
        super().__init__(**kwargs)
        self.only_return_final = only_return_final
        self.use_checkpoint = True

    def forward(self, images: torch.Tensor, return_dino_output: bool = False) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        if self.use_checkpoint:
            patch_tokens = checkpoint(self.patch_embed, images, use_reentrant=False)
        else:
            patch_tokens = self.patch_embed(images)

        dino_results = {}
        if isinstance(patch_tokens, dict):
            if return_dino_output:
                dino_results['cls_token'] = patch_tokens['x_norm_clstoken']
                dino_results['reg_tokens'] = patch_tokens['x_norm_regtokens']
                dino_results['patch_tokens'] = patch_tokens['x_norm_patchtokens']
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if not self.only_return_final:
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)

        if self.only_return_final:
            # return only the final output
            concat_inter = torch.cat([frame_intermediates[-1], global_intermediates[-1]], dim=-1)
            output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if return_dino_output:
            return output_list, self.patch_start_idx, dino_results
        else:
            return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if self.use_checkpoint:
            return checkpoint(super()._process_frame_attention, tokens, B, S, P, C, frame_idx, pos, use_reentrant=False)
        else:
            return super()._process_frame_attention(tokens, B, S, P, C, frame_idx, pos)

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if self.use_checkpoint:
            return checkpoint(super()._process_global_attention, tokens, B, S, P, C, global_idx, pos, use_reentrant=False)
        else:
            return super()._process_global_attention(tokens, B, S, P, C, global_idx, pos)