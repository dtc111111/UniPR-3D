import pytorch_lightning as pl

from vpr_vggt_ft import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

VGGT_MEAN_STD = {'mean': [0.0, 0.0, 0.0], 
                 'std': [1.0, 1.0, 1.0]}

if __name__ == '__main__':        
    datamodule = GSVCitiesDataModule(
        batch_size=15,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(392, 518),
        mean_std=VGGT_MEAN_STD,
        num_workers=4,
        show_data_stats=True,
        min_support_frames=0,
        max_support_frames=0,
        # val_set_names=['msls_seq_val'],
        val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
    )
    
    model = VPRModel(
        lr = 6e-5,
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
        gather_loss=True,

        vggt_aggregator_pretrain='model.pt',
        vggt_geo_salad_pretrain='./logs/lightning_logs/version_69/checkpoints/vggt_pose_(03)_R1[0.9095]_R5[0.9581].ckpt',
        vggt_dino_salad_pretrain='./logs/lightning_logs/version_69/checkpoints/vggt_pose_(03)_R1[0.9095]_R5[0.9581].ckpt',
        # vggt_geo_salad_pretrain='./logs/lightning_logs/version_41/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        # vggt_dino_salad_pretrain='./logs/lightning_logs/version_41/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        # vggt_camera_head_pretrain='model.pt',
        with_dinov2_features=True,
        with_geo_features=True,
        finetune_frame_attn=True,
        finetune_global_attn=False,
        finetune_patch_embed=True,
        finetune_aggregator_layers=1,
        finetune_patch_embed_layers=1,
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='msls_val/R1',
        filename='vggt_pose' + '_({epoch:02d})_R1[{msls_val/R1:.4f}]_R5[{msls_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode='max'
    )

    # spf_callback = SupportFrameDatasetCallback(datamodule)
    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        # strategy='ddp',
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=4,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        sync_batchnorm=True,
    )
    # Run validation before training
    # trainer.validate(model=model, datamodule=datamodule)
    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
