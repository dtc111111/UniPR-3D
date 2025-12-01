import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim import lr_scheduler, optimizer
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
import torch.nn as nn
import torch.nn.functional as F

import utils

from vggt.models.vggtpr_lora import VGGTPR_LoRA


def load_pretrain(pretrain_path):
    weights = torch.load(pretrain_path, map_location='cpu', weights_only=False)
    if 'model' in weights:
        weights = weights['model']
    if 'state_dict' in weights:
        weights = weights['state_dict']
    if len(weights) > 0 and next(iter(weights.keys())).startswith('model.'):
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
    return weights


def normalize_3dof_pose(pose):
    """
    归一化3DoF相机位姿
    Args:
        pose: (B_all, S, 3), 3维分别为: x, y, northdeg
    Returns:
        normalized_pose: (B_all, S, 3), 归一化后的相机位姿
    """
    # scale using xy translation norm
    dist = pose[..., :2].norm(dim=-1)  # [B, S]
    dist_norm = dist.mean(dim=1) # [B]
    dist_norm = dist_norm.clamp(min=1e-6, max=1e6)
    pose_new = pose.clone()
    pose_new[..., :2] = pose[..., :2] / dist_norm.view(-1, 1, 1)

    return pose_new


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        vggt_aggregator_pretrain=None,
        vggt_geo_salad_pretrain=None,
        vggt_dino_salad_pretrain=None,
        vggt_camera_head_pretrain=None,
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        # loss_name='MultiSimilarityLoss', 
        # miner_name='MultiSimilarityMiner', 
        # miner_margin=0.1,
        triplet_margin=0.1,
        faiss_gpu=False,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        with_camera_pose=False,
        camera_pose_type='yaw', # 'yaw' or '3dof'
        with_dinov2_features=False,
        lora_frame_attn=True,
        lora_global_attn=True,
        lora_patch_embed=False,
        online_mining=False,
    ):
        super().__init__()

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args
        self.with_camera_pose = with_camera_pose
        self.camera_pose_type = camera_pose_type

        # Loss
        # self.loss_name = loss_name
        # self.miner_name = miner_name
        # self.miner_margin = miner_margin
        self.triplet_margin = triplet_margin
        self.online_mining = online_mining
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.criterion_triplet = nn.TripletMarginLoss(margin=self.triplet_margin, p=2, reduction="sum")
        # self.loss_fn = utils.get_loss(loss_name)
        # self.miner = utils.get_miner(miner_name, miner_margin)
        # self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        print('loading model ...')
        self.model = VGGTPR_LoRA(lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_patch_embed=lora_patch_embed,
                                 lora_frame_attn=lora_frame_attn, lora_global_attn=lora_global_attn, with_dinov2_features=with_dinov2_features, 
                                 with_camera_pose=with_camera_pose, camera_pose_type=camera_pose_type)
        unique_pretrain_files = dict()
        if vggt_aggregator_pretrain is not None:
            print('loading aggregator weights from:', vggt_aggregator_pretrain)
            aggregator_weights = load_pretrain(vggt_aggregator_pretrain)
            unique_pretrain_files[vggt_aggregator_pretrain] = aggregator_weights
            partial_weights = {k.replace("aggregator.", ""): v for k, v in aggregator_weights.items() if k.startswith('aggregator.')}
            missing_keys, unexpected_keys = self.model.aggregator.load_state_dict(partial_weights, strict=False)
            # filter out missing_keys related to LoRA
            missing_keys = [k for k in missing_keys if not k.endswith('.lora_A') and not k.endswith('.lora_B') and not k.endswith('.lora_dropout')]
            print(f'Warning: missing keys when loading pretrained weights for aggregator: {missing_keys}')
            print(f'Warning: unexpected keys when loading pretrained weights for aggregator: {unexpected_keys}')

        if vggt_geo_salad_pretrain is not None:
            print('loading geo salad head weights from:', vggt_geo_salad_pretrain)
            if vggt_geo_salad_pretrain not in unique_pretrain_files:
                salad_weights = load_pretrain(vggt_geo_salad_pretrain)
                unique_pretrain_files[vggt_geo_salad_pretrain] = salad_weights
            else:
                salad_weights = unique_pretrain_files[vggt_geo_salad_pretrain]
            partial_weights = {k.replace("geo_salad_head.", ""): v for k, v in salad_weights.items() if k.startswith('geo_salad_head.')}
            missing_keys, unexpected_keys = self.model.geo_salad_head.load_state_dict(partial_weights, strict=False)
            print(f'Warning: missing keys when loading pretrained weights for geo salad head: {missing_keys}')
            print(f'Warning: unexpected keys when loading pretrained weights for geo salad head: {unexpected_keys}')

        if vggt_dino_salad_pretrain is not None:
            print('loading dino salad head weights from:', vggt_dino_salad_pretrain)
            if vggt_dino_salad_pretrain not in unique_pretrain_files:
                salad_weights = load_pretrain(vggt_dino_salad_pretrain)
                unique_pretrain_files[vggt_dino_salad_pretrain] = salad_weights
            else:
                salad_weights = unique_pretrain_files[vggt_dino_salad_pretrain]
            partial_weights = {k.replace("dino_salad_head.", ""): v for k, v in salad_weights.items() if k.startswith('dino_salad_head.')}
            missing_keys, unexpected_keys = self.model.dino_salad_head.load_state_dict(partial_weights, strict=False)
            print(f'Warning: missing keys when loading pretrained weights for dino salad head: {missing_keys}')
            print(f'Warning: unexpected keys when loading pretrained weights for dino salad head: {unexpected_keys}')

        if vggt_camera_head_pretrain is not None:
            print('loading camera head weights from:', vggt_camera_head_pretrain)
            if vggt_camera_head_pretrain not in unique_pretrain_files:
                camera_head_weights = load_pretrain(vggt_camera_head_pretrain)
                unique_pretrain_files[vggt_camera_head_pretrain] = camera_head_weights
            else:
                camera_head_weights = unique_pretrain_files[vggt_camera_head_pretrain]
            partial_weights = {k.replace("camera_head.", ""): v for k, v in camera_head_weights.items() if k.startswith('camera_head.trunk')}
            missing_keys, unexpected_keys = self.model.camera_head.load_state_dict(partial_weights, strict=False)
            missing_keys = [k for k in missing_keys if not k.endswith('.lora_A') and not k.endswith('.lora_B') and not k.endswith('.lora_dropout')]
            print(f'Warning: missing keys when loading pretrained weights for camera head: {missing_keys}')
            print(f'Warning: unexpected keys when loading pretrained weights for camera head: {unexpected_keys}')

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x, camera_pose=None):
        # assert False, "Please use the model in eval or train mode"
        pred_dict = self.model(x, camera_pose=camera_pose)
        x = pred_dict['salad_pred'] # B, C_out
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        

        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]
    
    
    def on_fit_start(self):
        self.trainer.datamodule.model_cache = self
        # print('device in setup:', self.device)
        self.trainer.datamodule.fit_setup()

    # def setup(self, stage):
    #     self.trainer.datamodule.model_cache = self
    #     print('device in setup:', self.device)
    #     self.trainer.datamodule.msls_setup(stage)

    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        
    #  The loss function call (this method will be called at each training iteration)
    # def loss_function(self, descriptors_local, labels_local):
    #     # we mine the pairs/triplets if there is an online mining strategy
    #     # print('loss_shape:', descriptors_local.shape)
    #     if self.gather_loss:
    #         gathered_descriptors = self.all_gather(descriptors_local, sync_grads=True)
    #         gathered_labels = self.all_gather(labels_local, sync_grads=True)
    #         descriptors = gathered_descriptors.flatten(0,1)
    #         labels = gathered_labels.flatten(0,1)
    #         # print('gathered_loss:', descriptors.shape)
    #     else:
    #         descriptors = descriptors_local
    #         labels = labels_local

    #     if self.miner is not None:
    #         miner_outputs = self.miner(descriptors, labels)
    #         loss = self.loss_fn(descriptors, labels, miner_outputs)
            
    #         # calculate the % of trivial pairs/triplets 
    #         # which do not contribute in the loss value
    #         nb_samples = descriptors.shape[0]
    #         nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
    #         batch_acc = 1.0 - (nb_mined/nb_samples)

    #     else: # no online mining
    #         loss = self.loss_fn(descriptors, labels)
    #         batch_acc = 0.0
    #         if type(loss) == tuple: 
    #             # somes losses do the online mining inside (they don't need a miner objet), 
    #             # so they return the loss and the batch accuracy
    #             # for example, if you are developping a new loss function, you might be better
    #             # doing the online mining strategy inside the forward function of the loss class, 
    #             # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
    #             loss, batch_acc = loss

    #     # keep accuracy of every batch and later reset it at epoch start
    #     self.batch_acc.append(batch_acc)
    #     # log it
    #     self.log('b_acc', sum(self.batch_acc) /
    #             len(self.batch_acc), prog_bar=True, logger=True)
    #     # print('local_loss:', loss)
    #     return loss
    
    # def on_train_batch_start(self, batch, batch_idx):
        
    #     if isinstance(self.trainer.datamodule, GSVCitiesDataModule):
    #         self.trainer.datamodule.update_support_frames_num()
    #         # print(f"Updated support_frames_num to: {new_support_frames_num}")

    # This is the training step that's executed at each iteration


    def mean_triplet_loss(self, descriptors):
        loss_triplet = 0
        nNeg = descriptors.shape[1] - 2  # number of negatives per triplet # one query one positive
        B = descriptors.shape[0]
        for b in range(B):
            anchor = descriptors[b:b+1, 0]  # Anchor descriptor
            positive = descriptors[b:b+1, 1]  # Positive descriptor
            negatives = descriptors[b, 2:]  # Negative descriptors


            loss_triplet += self.criterion_triplet(anchor, positive, negatives)
        loss_triplet /= B * nNeg
        return loss_triplet
    
    def online_mining_triplet_loss(self, descriptors, strategy='semi_hard'):
        loss_triplet = 0
        nPos = self.trainer.datamodule.nPos
        nNeg = self.trainer.datamodule.nNeg
        assert descriptors.shape[1] == 1 + nPos + nNeg, "descriptors shape does not match nPos and nNeg"
        B = descriptors.shape[0]
    
        # 1. 分割query, positives, negatives
        queries = descriptors[:, 0, :]  # (B, C)
        positives = descriptors[:, 1:1+nPos, :]  # (B, nPos, C)
        negatives = descriptors[:, 1+nPos:, :]  # (B, nNeg, C)
        
        # 2. 计算query到所有样本的距离
        queries_exp = queries.unsqueeze(1)  # (B, 1, C)
        
        # 计算query到正样本的距离
        dist_pos = torch.cdist(queries_exp, positives, p=2).squeeze(1)  # (B, nPos)
        
        # 计算query到负样本的距离
        dist_neg = torch.cdist(queries_exp, negatives, p=2).squeeze(1)  # (B, nNeg)
        
        # 3. 为每个query选择最难的正样本（距离最远的）
        hardest_pos_dist, _ = torch.max(dist_pos, dim=1)  # (B,)

        if strategy == 'semi_hard':
            # 半难例挖掘策略
            # 扩展维度用于广播
            d_ap_exp = hardest_pos_dist.unsqueeze(1)  # (B, 1)
            
            # 创建半难例条件掩码：d_ap < d_an < d_ap + margin
            semi_hard_mask = (dist_neg > d_ap_exp) & (dist_neg < d_ap_exp + self.triplet_margin)
            
            # 将不满足条件的距离设为无穷大
            dist_neg_masked = dist_neg.clone()
            dist_neg_masked[~semi_hard_mask] = float('inf')
            
            # 找到每个query的半难负样本的最小距离
            d_an, _ = torch.min(dist_neg_masked, dim=1)
            
            # 检查哪些query没有半难负样本
            no_semi_hard = torch.isinf(d_an)
            
            if no_semi_hard.any():
                # 对没有半难负样本的query使用难例挖掘
                d_an_hard, _ = torch.min(dist_neg, dim=1)
                d_an[no_semi_hard] = d_an_hard[no_semi_hard]

        elif strategy == 'hard':
            # 难例挖掘：选择最难负样本
            d_an, _ = torch.min(dist_neg, dim=1)
        # elif strategy == 'all':
        #     # 使用所有正负样本组合
        #     # 扩展维度
        #     dist_pos_exp = dist_pos.unsqueeze(2)  # (B, nPos, 1)
        #     dist_neg_exp = dist_neg.unsqueeze(1)  # (B, 1, nNeg)
            
        #     # 计算所有组合的损失
        #     all_losses = F.relu(dist_pos_exp - dist_neg_exp + self.triplet_margin)  # (B, nPos, nNeg)
            
        #     # 选择最难的三元组（最大损失）
        #     hardest_losses, _ = torch.max(all_losses.view(B, -1), dim=1)
        #     loss_triplet = hardest_losses.mean()
        #     return loss_triplet
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        losses = F.relu(hardest_pos_dist - d_an + self.triplet_margin)
        loss_triplet = losses.mean()
        return loss_triplet


    def training_step(self, batch, batch_idx):
        images, camera_pose = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places

        # reshape places and labels
        # if len(places.shape) == 5:
        #     BS, N, ch, h, w = places.shape
        #     images = places.view(BS*N, 1, ch, h, w)
        #     camera_pose = camera_pose.view(BS*N, 1, 3)
        # elif len(places.shape) == 6:
        #     BS, N, S_all, ch, h, w = places.shape
        #     images = places.view(BS*N, S_all, ch, h, w)
        #     camera_pose = camera_pose.view(BS*N, S_all, 3)
        # N = query + positive + negatives
        B, N, S, ch, h, w = images.shape
        images = images.view(B*N, S, ch, h, w)
        # print(images.shape)
        camera_pose = camera_pose.view(B*N, S, 3)

        camera_pose = normalize_3dof_pose(camera_pose)
        if self.camera_pose_type == 'yaw':
            camera_pose = camera_pose[..., 2] # only keep the yaw angle
        # print('input_shape:', images.shape)
        # print('camera_pose.shape:', camera_pose.shape)
        # labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images, camera_pose) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')
        
        descriptors = descriptors.view(B, N, -1)  # B, N, C
        if self.online_mining:
            device_loss_triplet = self.online_mining_triplet_loss(descriptors)
        else:
            device_loss_triplet = self.mean_triplet_loss(descriptors)

        del images, camera_pose

        # gathered_loss_triplet = self.all_gather(device_loss_triplet, sync_grads=True)
        # print('gathered_loss_triplet:', gathered_loss_triplet)
        # loss_triplet = gathered_loss_triplet.mean()

        # loss_pr = loss_triplet
        loss_pr = device_loss_triplet
        loss = loss_pr

        # loss_pr = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        # loss = loss_pr
        # if self.with_camera_pose:
        #     loss_cam_pose = utils.compute_camera_loss_3dof(
        #         pred_pose_encodings=camera_pose_list,
        #         gt_pose_encoding=camera_pose)
        #     loss += loss_cam_pose['loss_camera']

        self.log('loss_pr', loss_pr.item(), logger=True, prog_bar=True)
        # if self.with_camera_pose:
        #     self.log('loss_cam_pose', loss_cam_pose['loss_camera'].item(), logger=True, prog_bar=True)
        return {'loss': loss}

    # def on_train_epoch_end(self):
    #     # we empty the batch_acc list for next epoch
    #     self.batch_acc = []
    #     self.trainer.strategy.barrier()
    #     self.trainer

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # places, index = batch
        if len(batch) == 2:
            places, index = batch
            BS, ch, h, w = places.shape
            images = places.view(BS, 1, ch, h, w)
            camera_pose = torch.zeros((BS, 1, 3)).to(places.device) # camera poses for anchor frames are all zeros
        elif len(batch) == 3:
            images, camera_pose, index = batch
            BS, S_all, ch, h, w = images.shape
            # images = places.view(BS, S_all, ch, h, w)
            # camera_pose = camera_pose.view(BS, S_all, 3)
            assert camera_pose.shape == (BS, S_all, 3)
        else:
            raise ValueError('Validation batch should contain 2 or 3 elements')
        
        camera_pose = normalize_3dof_pose(camera_pose)
        if self.camera_pose_type == 'yaw':
            camera_pose = camera_pose[..., 2] # only keep the yaw angle

        descriptors = self(images, camera_pose)
        res = {
            'descriptors': descriptors.detach().cpu(),
            'index': index.detach().cpu(),
        }
        self.val_outputs[dataloader_idx].append(res)
        return res['descriptors']
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
        
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        # val_step_outputs = self.val_outputs
        val_step_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
        for i in range(len(self.trainer.datamodule.val_datasets)):
            for output in self.val_outputs[i]:
                gathered_descriptors = self.all_gather(output['descriptors']).permute(1,0,2).reshape(-1, output['descriptors'].shape[-1]).cpu()
                gathered_index = self.all_gather(output['index']).permute(1,0).reshape(-1).cpu()
                val_step_outputs[i].append((gathered_index, gathered_descriptors))
        
        # Compute metrics
        # if self.trainer.is_global_zero:
        val_step_final = []
        for i in range(len(self.trainer.datamodule.val_datasets)):
            concat_descriptors = torch.cat([desc for idx, desc in val_step_outputs[i]], dim=0)
            concat_index = torch.cat([idx for idx, desc in val_step_outputs[i]], dim=0)
            # print(concat_index)
            valid_length = len(self.trainer.datamodule.val_datasets[i])
            concat_descriptors = concat_descriptors[:valid_length]
            concat_index = concat_index[:valid_length]
            # sort according to the index
            val_step_final.append(concat_descriptors)
        
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        # if len(dm.val_datasets)==1: # we need to put the outputs in a list
        #     val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            # feats = torch.concat(val_step_outputs[i], dim=0)
            feats = val_step_final[i]
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'nordland_seq' in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'robotcar_seq' in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []


def print_trainable_parameters(model):
    total_params = 0
    print("=" * 80)
    print("模型可训练参数详情:")
    print("=" * 80)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name:<60} | 形状: {str(param.shape):<20} | 参数数量: {param_count:>10,}")
    
    print("=" * 80)
    print(f"总可训练参数数量: {total_params:,}")
    print("=" * 80)
    
    return total_params

if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = VPRModel(
        lr = 1e-5,
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        triplet_margin=0.1,
        faiss_gpu=False,
        vggt_aggregator_pretrain='camera_pose_yaw.pth',
        vggt_geo_salad_pretrain='./logs/lightning_logs/version_41/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        vggt_dino_salad_pretrain='./logs/lightning_logs/version_41/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        # vggt_camera_head_pretrain='model.pt',
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_frame_attn=True,
        lora_global_attn=True,
        lora_patch_embed=True,
        with_camera_pose=True,
        camera_pose_type='yaw',
        with_dinov2_features=True,
    ).cuda()
    print_trainable_parameters(model)

    x = torch.randn(3, 3, 3, 392, 518).cuda()
    camera_pose = torch.randn(3, 3).cuda()
    y = model(x, camera_pose)
    print(y[0].shape)