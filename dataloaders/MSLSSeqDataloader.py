import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader, RandomSampler
from torch.utils.data import Subset
from torchvision import transforms as T
import multiprocessing as mp
# from dataloaders.GSVCitiesDataset import GSVCitiesDataset
# from dataloaders.SFrameBatchSampler import SFrameBatchSampler
from dataloaders.MapillarySeqTrainDataset import TrainingMSLS_seq, TrainingMSLSNoMining_seq
from dataloaders.NordlandSeqDataset import NordlandSeqDataset
from dataloaders.RobotCarSeqDataset import RobotCarSeqDataset
from . import PittsburgDataset
from . import MapillaryDataset
from . import MapillarySeqDataset
import numpy as np
from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

# VGGT uses no normalization
# to be consistent with the original VGG16 training procedure
VGGT_MEAN_STD = {'mean': [0.0, 0.0, 0.0], 
                 'std': [1.0, 1.0, 1.0]}


class MSLSSeqDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 shuffle_all=True,
                 image_size=(392, 518),
                 num_workers=4,
                 show_data_stats=True,
                 mean_std=VGGT_MEAN_STD,
                 feature_dim=17152,
                 cache_queries=1000,
                 cache_negatives=1000,
                 nNeg=10,
                 nPos=4,
                 msls_seq_len=5,
                 mining_type='online', # 'offline' or 'online'
                 using_subset=False,
                 subset_size=1000,
                 cities=['melbourne'],
                 val_set_names=['pitts30k_val', 'msls_val']
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.show_data_stats = show_data_stats
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.val_set_names = val_set_names
        self.feature_dim = feature_dim
        self.cached_queries = cache_queries
        self.cached_negatives = cache_negatives
        self.msls_seq_len = msls_seq_len
        self.nNeg = nNeg
        self.nPos = nPos
        self.model_cache = None
        self.using_subset = using_subset
        self.subset_size = subset_size
        self.mining_type = mining_type
        self.cities = cities
        # if self.max_support_frames > 0:
        #     self.num_workers_train = 0
        # else:
        #     self.num_workers_train = num_workers
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            # 'drop_last': False,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'shuffle': self.shuffle_all
        }

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage):
        print('Setting up val dataloaders ...')
        # load validation sets (pitts_val, msls_val, ...etc)
        self.val_datasets = []
        for valid_set_name in self.val_set_names:
            if valid_set_name.lower() == 'pitts30k_test':
                self.val_datasets.append(PittsburgDataset.get_whole_test_set(
                    input_transform=self.valid_transform))
            elif valid_set_name.lower() == 'pitts30k_val':
                self.val_datasets.append(PittsburgDataset.get_whole_val_set(
                    input_transform=self.valid_transform))
            elif valid_set_name.lower() == 'msls_val':
                self.val_datasets.append(MapillaryDataset.MSLS(
                    input_transform=self.valid_transform))
            elif valid_set_name.lower() == 'msls_seq_val':
                self.val_datasets.append(MapillarySeqDataset.MSLS_seq(
                    input_transform=self.valid_transform, redis=True))
            elif valid_set_name.lower() == 'nordland_seq_2':
                self.val_datasets.append(NordlandSeqDataset(
                    pos_thresh=5, neg_thresh=5,
                    input_transform=self.valid_transform))
            elif valid_set_name.lower() == 'nordland_seq':
                self.val_datasets.append(NordlandSeqDataset(
                    input_transform=self.valid_transform))
            elif valid_set_name.lower() == 'robotcar_seq':
                self.val_datasets.append(RobotCarSeqDataset(
                    input_transform=self.valid_transform))
            elif valid_set_name.lower() == 'robotcar_seq_2':
                self.val_datasets.append(RobotCarSeqDataset(
                    input_transform=self.valid_transform,
                    root_dir='/nas0/dataset/vggt-pr_extra_datasets/oxford2'))
            else:
                print(
                    f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                raise NotImplementedError
            
    def fit_setup(self):
        self.reload()
        self.init_reload = True
        if self.show_data_stats:
            self.print_stats()

    def reload(self):
        if self.mining_type == 'offline':
            self.train_dataset = TrainingMSLS_seq(
                cities=self.cities,
                seq_length=self.msls_seq_len,
                cached_queries=self.cached_queries,
                cached_negatives=self.cached_negatives,
                nNeg=self.nNeg,
                features_dim=self.feature_dim,
                input_transform=self.train_transform)
            print('Computing training dataset cache ...')
            self.train_dataset.compute_triplets(self.model_cache)
        elif self.mining_type == 'online':
            self.train_dataset = TrainingMSLSNoMining_seq(
                cities=self.cities,
                seq_length=self.msls_seq_len,
                nNeg=self.nNeg,
                nPos=self.nPos,
                input_transform=self.train_transform)
            if self.using_subset:
                self.train_dataset = Subset(self.train_dataset,
                                            np.random.choice(len(self.train_dataset), self.subset_size, replace=False))
        else:
            raise NotImplementedError(f'Mining type {self.mining_type} not implemented yet')
        
        
    # def update_support_frames_num(self):
    #     new_support_frames_num = np.random.randint(self.min_support_frames, self.max_support_frames + 1)
    #     self.train_dataset.support_frames_num = new_support_frames_num
    #     # print(f'Updated support frames number to: {new_support_frames_num}')

    def train_dataloader(self):
        if self.init_reload:
            self.init_reload = False
        else:
            self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()  # print a new line
        # table = PrettyTable()
        # table.field_names = ['Data', 'Value']
        # table.align['Data'] = "l"
        # table.align['Value'] = "l"
        # table.header = False
        # # table.add_row(["# of cities", f"{len(TRAIN_CITIES)}"])
        # # table.add_row(["# of seqs", f'{self.train_dataset.get_total_len()}'])
        # print(table.get_string(title="Training Dataset"))
        # print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size", f"{self.batch_size}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__() // self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
