import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim import lr_scheduler, optimizer
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

import utils

from vggt.models.vggtpr_ft import VGGTPR_Finetune


def load_pretrain(pretrain_path):
    weights = torch.load(pretrain_path, map_location='cpu', weights_only=False)
    if 'model' in weights:
        weights = weights['model']
    if 'state_dict' in weights:
        weights = weights['state_dict']
    if len(weights) > 0 and next(iter(weights.keys())).startswith('model.'):
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
    return weights


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
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False,
        gather_loss=False,
        with_dinov2_features=False,
        with_geo_features=True,
        finetune_patch_embed=False,
        finetune_global_attn=True,
        finetune_frame_attn=True,
        finetune_patch_embed_layers=-1,
        finetune_aggregator_layers=-1,
    ):
        super().__init__()

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        self.gather_loss = gather_loss
        
        print('loading model ...')
        self.model = VGGTPR_Finetune(finetune_patch_embed=finetune_patch_embed,finetune_frame_attn=finetune_frame_attn, finetune_global_attn=finetune_global_attn, 
                                     with_dinov2_features=with_dinov2_features, with_geo_features=with_geo_features,
                                     finetune_aggregator_layers=finetune_aggregator_layers, finetune_patch_embed_layers=finetune_patch_embed_layers)
        unique_pretrain_files = dict()
        if vggt_aggregator_pretrain is not None:
            print('loading aggregator weights from:', vggt_aggregator_pretrain)
            aggregator_weights = load_pretrain(vggt_aggregator_pretrain)
            unique_pretrain_files[vggt_aggregator_pretrain] = aggregator_weights
            partial_weights = {k.replace("aggregator.", ""): v for k, v in aggregator_weights.items() if k.startswith('aggregator.')}
            missing_keys, unexpected_keys = self.model.aggregator.load_state_dict(partial_weights, strict=False)
            # filter out missing_keys related to LoRA
            print(f'Warning: missing keys when loading pretrained weights for aggregator: {missing_keys}')
            print(f'Warning: unexpected keys when loading pretrained weights for aggregator: {unexpected_keys}')

        if vggt_geo_salad_pretrain is not None:
            print('loading geo salad head weights from:', vggt_geo_salad_pretrain)
            if vggt_geo_salad_pretrain not in unique_pretrain_files:
                salad_weights = load_pretrain(vggt_geo_salad_pretrain)
                unique_pretrain_files[vggt_geo_salad_pretrain] = salad_weights
            else:
                salad_weights = unique_pretrain_files[vggt_geo_salad_pretrain]
            # partial_weights = {k.replace("salad_head.", ""): v for k, v in salad_weights.items() if k.startswith('salad_head.')}
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
    def forward(self, x):
        # assert False, "Please use the model in eval or train mode"
        pred_dict = self.model(x)
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
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors_local, labels_local):
        # we mine the pairs/triplets if there is an online mining strategy
        # print('loss_shape:', descriptors_local.shape)
        if self.gather_loss:
            gathered_descriptors = self.all_gather(descriptors_local, sync_grads=True)
            gathered_labels = self.all_gather(labels_local, sync_grads=True)
            descriptors = gathered_descriptors.flatten(0,1)
            labels = gathered_labels.flatten(0,1)
            # print('gathered_loss:', descriptors.shape)
        else:
            descriptors = descriptors_local
            labels = labels_local

        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        # print('local_loss:', loss)
        return loss
    
    # def on_train_batch_start(self, batch, batch_idx):
        
    #     if isinstance(self.trainer.datamodule, GSVCitiesDataModule):
    #         self.trainer.datamodule.update_support_frames_num()
    #         # print(f"Updated support_frames_num to: {new_support_frames_num}")

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, _, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places

        # reshape places and labels
        if len(places.shape) == 5:
            BS, N, ch, h, w = places.shape
            images = places.view(BS*N, 1, ch, h, w)
        elif len(places.shape) == 6:
            BS, N, S_all, ch, h, w = places.shape
            images = places.view(BS*N, S_all, ch, h, w)

        # print('input_shape:', images.shape)
        # print('camera_pose.shape:', camera_pose.shape)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')

        # disable AMP for loss
        with torch.amp.autocast('cuda', enabled=False):
            loss_pr = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        loss = loss_pr
        # if self.with_camera_pose:
        #     loss_cam_pose = utils.compute_camera_loss_3dof(
        #         pred_pose_encodings=camera_pose_list,
        #         gt_pose_encoding=camera_pose)
        #     loss += loss_cam_pose['loss_camera']

        self.log('loss_pr', loss_pr.item(), logger=True, prog_bar=True)
        # if self.with_camera_pose:
        #     self.log('loss_cam_pose', loss_cam_pose['loss_camera'].item(), logger=True, prog_bar=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # places, index = batch
        if len(batch) == 2:
            places, index = batch
            BS, ch, h, w = places.shape
            images = places.view(BS, 1, ch, h, w)
        elif len(batch) == 3:
            images, _, index = batch
            BS, S_all, ch, h, w = images.shape
        else:
            raise ValueError('Validation batch should contain 2 or 3 elements')
        

        descriptors = self(images)
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


# class SupportFrameDatasetCallback(pl.Callback):
#     def __init__(self, datamodule:GSVCitiesDataModule, min_support_frames=0, max_support_frames=5):
#         super().__init__()
#         self.dataset = datamodule
#         self.min_support_frames = min_support_frames
#         self.max_support_frames = max_support_frames
    
#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#         # 在每个训练batch开始前更新数据集参数
#         new_support_frames_num = np.random.randint(self.min_support_frames, self.max_support_frames + 1)
#         self.dataset.train_dataset.set_support_frames_num(new_support_frames_num)
#         # print(f"Updated support_frames_num to: {new_support_frames_num}")


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
        gather_loss=True, # gather descriptors from all gpus for loss and mining

        vggt_aggregator_pretrain='model.pt',
        vggt_geo_salad_pretrain='./logs/lightning_logs/version_41_single_ckpt/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        vggt_dino_salad_pretrain='./logs/lightning_logs/version_41_single_ckpt/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        # vggt_camera_head_pretrain='model.pt',
        finetune_patch_embed=False,
        finetune_global_attn=False,
        finetune_frame_attn=False,
        with_dinov2_features=True,
        with_geo_features=True,
    ).cuda()
    print_trainable_parameters(model)

    x = torch.randn(3, 3, 3, 392, 518).cuda()
    y = model(x)
    print(y[0].shape)