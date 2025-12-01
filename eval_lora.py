import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse


from vpr_vggt_lora import VPRModel
# from vpr_vggt import VPRModel as VPRModelNoPose
from utils.validation import get_validation_recalls
from utils.resize import ResizeToGivenSize, ResizeWithPadding
# Dataloader
from dataloaders.val.NordlandDataset import NordlandDataset
from dataloaders.val.MapillaryDataset import MSLS
from dataloaders.val.MapillaryTestDataset import MSLSTest
from dataloaders.RobotCarSeqDataset import RobotCarSeqDataset
from dataloaders.val.PittsburghDataset import PittsburghDataset
from dataloaders.val.SPEDDataset import SPEDDataset
from dataloaders.MapillarySeqDataset import MSLS_seq
from dataloaders.NordlandSeqDataset import NordlandSeqDataset

# VAL_DATASETS = ['MSLS', 'MSLS_Test', 'pitts30k_test', 'pitts250k_test', 'SPED', 'NordLand', 'MSLS_seq']
# VAL_DATASETS = ['MSLS', 'NordLand', 'pitts30k_test', 'pitts250k_test', 'SPED', ]
#VAL_DATASETS = ['SPED', 'NordLand', 'pitts250k_test']
#VAL_DATASETS = ['MSLS', 'MSLS_Test', 'pitts30k_test']
# VAL_DATASETS = ['NordLand_seq', 'MSLS_seq', 'RobotCar', 'RobotCar_2']
VAL_DATASETS = ['RobotCar']

# EVAL_DISTANCE = 25 # 2, 5, 10, 15, 20, 25 meters
# EVAL_NORDLAND_FRAMES_GAP = 25 # 1, 5, 10, 15, 20, 25 # in number of frames
# EVAL_NORDLAND_GAP = 2.4
SEQ_LENGTH = 5

def input_transform(image_size=None):
    MEAN=[0.0, 0.0, 0.0]; STD=[1.0, 1.0, 1.0]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            # ResizeToGivenSize(image_size),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

def get_val_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)
    
    if 'nordland_seq' in dataset_name:
        ds = NordlandSeqDataset(input_transform=transform,
                                seq_len=SEQ_LENGTH,
                                # pos_thresh=EVAL_NORDLAND_GAP * (EVAL_NORDLAND_FRAMES_GAP + 0.5), 
                                # neg_thresh=EVAL_NORDLAND_GAP * (EVAL_NORDLAND_FRAMES_GAP + 0.5)
                                )
    elif 'nordland' in dataset_name:    
        ds = NordlandDataset(input_transform=transform)
    elif 'msls_test' in dataset_name:
        ds = MSLSTest(input_transform=transform)
    elif 'msls_seq' in dataset_name:
        ds = MSLS_seq(input_transform=transform, seq_length=SEQ_LENGTH, redis=True,
                    #   pos_thresh=EVAL_DISTANCE, neg_thresh=EVAL_DISTANCE
                      )
    elif 'msls' in dataset_name:
        ds = MSLS(input_transform=transform)
    elif 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)
    elif 'sped' in dataset_name:
        ds = SPEDDataset(input_transform=transform)
    elif 'robotcar_2' in dataset_name:
        ds = RobotCarSeqDataset(input_transform=transform,
                                seq_len=SEQ_LENGTH, 
                                # pos_thresh=EVAL_DISTANCE, neg_thresh=EVAL_DISTANCE,
                                root_dir='/nas0/dataset/vggt-pr_extra_datasets/oxford2')
    elif 'robotcar' in dataset_name:
        ds = RobotCarSeqDataset(input_transform=transform,
                                seq_len=SEQ_LENGTH,
                                # pos_thresh=EVAL_DISTANCE, neg_thresh=EVAL_DISTANCE
                                )
    else:
        raise ValueError
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                # print('Batch length:', len(batch))
                if len(batch) == 2:
                    places, index = batch
                    BS, ch, h, w = places.shape
                    images = places.view(BS, 1, ch, h, w).to(device)
                    camera_pose = torch.zeros((BS, 1)).to(device) # camera poses for anchor frames are all zeros
                elif len(batch) == 3:
                    images, camera_pose, index = batch
                    BS, S_all, ch, h, w = images.shape
                    # images = places.view(BS, S_all, ch, h, w)
                    # camera_pose = camera_pose.view(BS, S_all, 3)
                    images = images.to(device)
                    # print(images.shape, camera_pose.shape)
                    camera_pose = camera_pose.to(device)
                    camera_pose = camera_pose[..., 2] # Use yaw only
                    # assert camera_pose.shape == (BS, S_all, 3)
                    assert camera_pose.shape == (BS, S_all)
                else:
                    raise ValueError('Validation batch should contain 2 or 3 elements')

                output = model(images, camera_pose) # Here we are calling the method forward that we defined above
                # output = model(images) # Here we are calling the method forward that we defined above
                descriptors.append(output.detach().cpu())

    return torch.cat(descriptors)

def load_model(ckpt_path):
    ckpt_path = '/home/vggt-pr/logs/lightning_logs/version_62_multi_ckpt/checkpoints/vggt_pose_(03)_R1[0.9365]_R5[0.9567].ckpt'
    model = VPRModel(
        vggt_aggregator_pretrain=ckpt_path,
        vggt_geo_salad_pretrain=ckpt_path,
        vggt_dino_salad_pretrain=ckpt_path,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_frame_attn=True,
        lora_global_attn=True,
        lora_patch_embed=False,
        with_camera_pose=False,
        camera_pose_type='yaw',
        with_dinov2_features=True,
        with_geo_features=True,
    )
    # model = VPRModelNoPose(
    #     vggt_pretrained_pt=ckpt_path
    # )
    model = model.eval()
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint")
    
    # Datasets parameters
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )
    parser.add_argument('--image_size', nargs='*', default=(392, 518), help='Image size (int, tuple or None)')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (args.image_size[0], args.image_size[0])
        elif len(args.image_size) == 2:
            args.image_size = tuple(args.image_size)
        else:
            raise ValueError('Invalid image size, must be int, tuple or None')
        
        args.image_size = tuple(map(int, args.image_size))

    return args


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    # print('EVAL_DISTANCE:', EVAL_DISTANCE)
    # print('EVAL_NORDLAND_FRAMES_GAP:', EVAL_NORDLAND_FRAMES_GAP)
    print('SEQ_LENGTH:', SEQ_LENGTH)
    args = parse_args()
    
    model = load_model(args.ckpt_path)

    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, args.image_size)
        val_loader = DataLoader(val_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, 'cuda')
        
        print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[ : num_references]
        q_list = descriptors[num_references : ]

        print('total_size', descriptors.shape[0], num_queries + num_references)

        testing = isinstance(val_dataset, MSLSTest)

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
            testing=testing,
        )

        if testing:
            val_dataset.save_predictions(preds, args.ckpt_path + '.' + 'vggtpr_eval2' + '.preds.txt')

        del descriptors
        print('========> DONE!\n\n')

