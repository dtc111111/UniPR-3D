import pytorch_lightning as pl
from datetime import timedelta
from vpr_vggt_lora_multiframe import VPRModel
from dataloaders.MSLSSeqDataloader import MSLSSeqDataModule
from dataloaders.MapillarySeqDataset import default_cities
from pytorch_lightning.strategies import DDPStrategy
VGGT_MEAN_STD = {'mean': [0.0, 0.0, 0.0], 
                 'std': [1.0, 1.0, 1.0]}

if __name__ == '__main__':        
    datamodule = MSLSSeqDataModule(
        batch_size=3,
        image_size=(392, 518),
        mean_std=VGGT_MEAN_STD,
        num_workers=5,
        msls_seq_len=5,
        cache_queries=0,
        cache_negatives=0,
        nNeg=8,
        nPos=2,
        show_data_stats=True,
        val_set_names=['msls_seq_val'],
        mining_type='online',
        using_subset=True,
        subset_size=4000,
        cities=default_cities['train'],
        #cities=['melbourne'], # cities to use for training
        # val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], # pitts30k_val, pitts30k_test, msls_val
    )
    
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
        # loss_name='MultiSimilarityLoss',
        # miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        # miner_margin=0.1,
        triplet_margin=0.1,
        # vggt_pretrained_pt='/nas0/dataset/vggt-pr_extra_datasets/model.pt',
        # vggt_pretrained_pt='./logs/lightning_logs/version_17/checkpoints/best.ckpt',
        # vggt_pretrained_pt='./logs/lightning_logs/version_34/checkpoints/last.ckpt',
        # vggt_aggregator_pretrain='./logs/lightning_logs/version_17/checkpoints/best.ckpt',
        vggt_aggregator_pretrain='/home/model.pt',
        # vggt_aggregator_pretrain='camera_pose_yaw.pth',
        vggt_geo_salad_pretrain='../vggt-pr-thin/logs/lightning_logs/version_41/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        vggt_dino_salad_pretrain='../vggt-pr-thin/logs/lightning_logs/version_41/checkpoints/vggt_pose_(04)_R1[0.9135]_R5[0.9608].ckpt',
        # vggt_camera_head_pretrain='model.pt',
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_frame_attn=True,
        lora_global_attn=True,
        lora_patch_embed=False,
        with_camera_pose=False,
        camera_pose_type='yaw',
        with_dinov2_features=True,
        online_mining=True,
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='msls_seq_val/R1',
        filename='vggt_pose' + '_({epoch:02d})_R1[{msls_seq_val/R1:.4f}]_R5[{msls_seq_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode='max'
    )

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(minutes=600))
    # spf_callback = SupportFrameDatasetCallback(datamodule)
    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy=strategy,
        # strategy='ddp',
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=8,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        sync_batchnorm=True,
        accumulate_grad_batches=4,
        # use_distributed_sampler=False,
    )
    # Run validation before training
    # trainer.validate(model=model, datamodule=datamodule)
    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
