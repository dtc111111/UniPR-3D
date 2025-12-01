# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.layers_lora import BlockLoRA, vit_small_lora, vit_base_lora, vit_large_lora, vit_giant2_lora
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.models.relative_pose_embed import CameraPoseEmbedding3dof, CameraPoseEmbeddingYaw
from vggt.models.utils import freeze_module, unfreeze_module
from vggt.layers.block import Block
from functools import partial

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class AggregatorLoRA(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        # block_fn=BlockLoRA,
        lora_frame_attn=True,
        lora_global_attn=True,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        with_camera_pose=False,
        pose_embedding_ratio=0.5,
        pose_embedding_type='yaw', # 'yaw' or '3dof'
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_patch_embed_layers=-1, # last N layers to apply LoRA on patch embed, -1 means all layers
        lora_aggregator_layers=-1, # last N layers to apply LoRA on aggregator, -1 means all layers
        freeze_patch_embed=True,
        freeze_camera_pose_embed=True,
        lora_patch_embed=False,
        only_return_final=True,
    ):
        super().__init__()

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.pose_embedding_ratio = pose_embedding_ratio
        self.freeze_patch_embed = freeze_patch_embed
        self.lora_patch_embed = lora_patch_embed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.only_return_final = only_return_final
        self.lora_patch_embed_layers = lora_patch_embed_layers

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        frame_block_fn = partial(BlockLoRA, lora_rank=lora_rank, lora_alpha=lora_alpha, \
                                 lora_dropout=lora_dropout) if lora_frame_attn else Block

        lora_aggregator_layers = lora_aggregator_layers if lora_aggregator_layers >= 0 else depth
        self.frame_blocks = []
        for i in range(depth):
            cur_block_fn = frame_block_fn if i >= depth - lora_aggregator_layers else frame_block_fn
            self.frame_blocks.append(
                cur_block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
            )
        self.frame_blocks = nn.ModuleList(self.frame_blocks)

        global_block_fn = partial(BlockLoRA, lora_rank=lora_rank, lora_alpha=lora_alpha, \
                                  lora_dropout=lora_dropout) if lora_global_attn else Block
        self.global_blocks = []
        for i in range(depth):
            cur_block_fn = global_block_fn if i >= depth - lora_aggregator_layers else global_block_fn
            self.global_blocks.append(
                cur_block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
            )
        self.global_blocks = nn.ModuleList(self.global_blocks)

        assert not lora_patch_embed or freeze_patch_embed, "lora_patch_embed requires freeze_patch_embed to be True"

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False
        
        self.with_camera_pose = with_camera_pose
        if with_camera_pose:
            self.pose_embedding_type = pose_embedding_type
            if pose_embedding_type == 'yaw':
                self.camera_pose_embedding = CameraPoseEmbeddingYaw(embed_dim=embed_dim)
            elif pose_embedding_type == '3dof':
                self.camera_pose_embedding = CameraPoseEmbedding3dof(embed_dim=embed_dim)
            else:
                raise ValueError(f"Unknown pose_embedding_type: {pose_embedding_type}")

        lora.mark_only_lora_as_trainable(self)

        if with_camera_pose and not freeze_camera_pose_embed:
            unfreeze_module(self.camera_pose_embedding)
        
        if not self.freeze_patch_embed:
            unfreeze_module(self.patch_embed)

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            if self.lora_patch_embed:
                vit_models = {
                    "dinov2_vitl14_reg": vit_large_lora,
                    "dinov2_vitb14_reg": vit_base_lora,
                    "dinov2_vits14_reg": vit_small_lora,
                    "dinov2_vitg2_reg": vit_giant2_lora,
                }
            else:
                vit_models = {
                    "dinov2_vitl14_reg": vit_large,
                    "dinov2_vitb14_reg": vit_base,
                    "dinov2_vits14_reg": vit_small,
                    "dinov2_vitg2_reg": vit_giant2,
                }

            patch_embed_lora_kwargs = {}
            if self.lora_patch_embed:
                patch_embed_lora_kwargs = {
                    "lora_rank": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                    "lora_layers": self.lora_patch_embed_layers,
                }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
                **patch_embed_lora_kwargs
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, camera_pose: torch.Tensor = None, return_dino_output: bool = False) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            camera_pose (torch.Tensor): Camera pose information with shape [B, S, 3].
                Required if with_camera_pose is True. The 3 dimensions are:
                - [:2] = relative translation vector T (2D)
                - [2:3] = rotation as northdeg (1D)

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
        if self.freeze_patch_embed and not self.lora_patch_embed: # disable gradients for patch embed
            with torch.no_grad():
                patch_tokens = self.patch_embed(images)
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

        if self.with_camera_pose:
            if camera_pose is None:
                raise ValueError("camera_pose must be provided when with_camera_pose is True")
            if self.pose_embedding_type == '3dof' and camera_pose.shape != (B, S, 3):
                raise ValueError(f"Expected camera_pose shape (B, S, 3), got {camera_pose.shape}")
            if self.pose_embedding_type == 'yaw' and camera_pose.shape != (B, S):
                raise ValueError(f"Expected camera_pose shape (B, S), got {camera_pose.shape}")

            # Get camera pose embeddings and add to camera tokens
            camera_pose_tokens = self.camera_pose_embedding(camera_pose, (B, S))  # (B*S, 1, C)
            # print("camera_pose_tokens example:", camera_pose_tokens[3, 0,:20])
            camera_token = camera_token + camera_pose_tokens * self.pose_embedding_ratio

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
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
