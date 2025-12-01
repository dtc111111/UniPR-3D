# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import loralib as lora
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from vggt.heads.camera_head import CameraHead
from vggt.models.aggregator_lora import AggregatorLoRA
from vggt.heads.salad_head import SaladHead
from vggt.models.utils import freeze_module, unfreeze_module

class VGGTPR_LoRA(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, with_camera_head=False, camera_pose_type='yaw', # 'yaw' or '3dof'
                 with_camera_pose=False, with_geo_features=True, with_dinov2_features=False, 
                 lora_patch_embed=False, lora_global_attn=True, lora_frame_attn=True, lora_patch_embed_layers=-1, lora_aggregator_layers=-1,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1):

        super().__init__()

        self.aggregator = AggregatorLoRA(img_size=img_size, patch_size=patch_size, pose_embedding_type=camera_pose_type,
                                         embed_dim=embed_dim, with_camera_pose=with_camera_pose, lora_global_attn=lora_global_attn, 
                                         lora_frame_attn=lora_frame_attn, lora_patch_embed=lora_patch_embed, lora_patch_embed_layers=lora_patch_embed_layers,
                                         lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_aggregator_layers=lora_aggregator_layers)
        
        self.geo_salad_head = SaladHead(dim_in=2 * embed_dim, use_fisrt_token=False) if with_geo_features else None
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if with_camera_head else None
        self.dino_salad_head = SaladHead(dim_in=embed_dim) if with_dinov2_features else None

    def forward(self, images: torch.Tensor, camera_pose: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            camera_pose (torch.Tensor): Camera pose information with shape [B, S, 3].
                Required if with_camera_pose is True. The 3 dimensions are:
                - [:2] = relative translation vector T (2D)
                - [2:3] = rotation as northdeg (1D)

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 3] (from the last iteration)
                - salad_pred (torch.Tensor): SALAD descriptor with shape [B, C_out]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        B, S = images.shape[:2]

        if self.dino_salad_head is not None:
            aggregated_tokens_list, patch_start_idx, dino_results = self.aggregator(images, camera_pose=camera_pose, return_dino_output=True)
        else:
            aggregated_tokens_list, patch_start_idx = self.aggregator(images, camera_pose=camera_pose)

        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.geo_salad_head is not None:
                salad_output = self.geo_salad_head(
                    aggregated_tokens_list, patch_start_idx=patch_start_idx
                )
                predictions["salad_pred"] = salad_output

            if self.dino_salad_head is not None:
                # print('dinov2 cls shape:', dino_results['cls_token'].unsqueeze(1).shape)
                # print('dinov2 reg shape:', dino_results['reg_tokens'].shape)
                # print('dinov2 patch shape:', dino_results['patch_tokens'].shape)
                # exit(0)
                dino_aggregated_token = torch.cat([dino_results['cls_token'].unsqueeze(1), dino_results['reg_tokens'], dino_results['patch_tokens']], dim=1)
                BS, P, C = dino_aggregated_token.shape
                dino_aggregated_tokens = [dino_aggregated_token.view(B, S, P, C)]
                dino_salad_output = self.dino_salad_head(
                    dino_aggregated_tokens, patch_start_idx=self.aggregator.patch_embed.num_register_tokens + 1
                )
                # concatenate geo and dino salad features
                if "salad_pred" not in predictions:
                    predictions["salad_pred"] = dino_salad_output
                else:
                    predictions["salad_pred"] = torch.cat([predictions["salad_pred"], dino_salad_output], dim=-1)

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

