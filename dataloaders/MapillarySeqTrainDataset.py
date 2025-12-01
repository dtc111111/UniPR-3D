import numpy as np
import torch
import os
import faiss
import hashlib
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from dataloaders.MapillarySeqDataset import MSLS_seq
from dataloaders.MapillarySeqDataset import default_cities


class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]


def encode_cities_to_md5(cities):
    """将城市列表编码为MD5字符串"""
    # 将城市列表转换为字符串（用逗号分隔）
    cities_str = ','.join(cities)
    
    # 创建MD5哈希对象
    md5_hash = hashlib.md5()
    
    # 更新哈希对象（需要将字符串编码为bytes）
    md5_hash.update(cities_str.encode('utf-8'))
    
    # 获取十六进制格式的MD5值
    return md5_hash.hexdigest()


class TrainingMSLS_seq(MSLS_seq):
    def __init__(self, root_dir=None, cities=['melbourne'], input_transform=None, task='seq2seq', seq_length=5, subtask='all', exclude_panos=True,
                 cached_queries=1000, cached_negatives=1000, nNeg=10, features_dim=4096):
        self.dataset_cache_dir = 'dataset_cache'
        os.makedirs(self.dataset_cache_dir, exist_ok=True)
        cities_str_md5 = encode_cities_to_md5(cities)
        self.cache_str = os.path.join(self.dataset_cache_dir, f'train_msls_{cities_str_md5}_seqlen_{seq_length}.pth')
        super().__init__(root_dir, cities, input_transform, task, seq_length, subtask, exclude_panos, cache_file=self.cache_str, pos_thresh=10, neg_thresh=25)
        print("Switching to training mode for MSLS_seq dataset...")
        self.cached_negatives = cached_negatives  # Number of negatives to randomly sample
        self.cached_queries = cached_queries      # Number of queries to randomly sample
        self.nNeg = nNeg                          # Number of negatives to use per query
        self.is_inference = False
        self.features_dim = features_dim
        self.qIdx = None
        self.pIdx = np.array(self.pIdx, dtype=object)
        self.compute_triplets_workers = 10
        self.compute_triplets_batch_size = 8
        # Additional initialization for training mode can be added here



    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)
        query_index, best_positive_index, neg_indexes = torch.split(self.triplets_global_indexes[index], (1, 1, self.nNeg))

        query_imgs, query_relative_poses, _ = super().__getitem__(query_index.item())
        positive_imgs,  positive_relative_poses, _ = super().__getitem__(best_positive_index.item())
        negative_imgs = []
        negative_relative_poses = []
        for neg_idx in neg_indexes:
            neg_imgs, neg_relative_poses, _ = super().__getitem__(neg_idx.item())
            negative_imgs.append(neg_imgs)
            negative_relative_poses.append(neg_relative_poses)
        # query = torch.stack(
        #     [self.base_transform(Image.open(join(self.dataset_folder, im))) for im in self.q_paths[query_index].split(',')])

        # positive = torch.stack(
        #     [self.base_transform(Image.open(join(self.dataset_folder, im))) for im in self.db_paths[best_positive_index].split(',')])

        # negatives = [torch.stack([self.base_transform(Image.open(join(self.dataset_folder, im)))for im in self.db_paths[idx].split(',')])
        #                 for idx in neg_indexes]

        images = torch.stack((query_imgs, positive_imgs, *negative_imgs), 0)
        # print('images shape:', images.shape)
        relative_poses = torch.stack((query_relative_poses, positive_relative_poses, *negative_relative_poses), 0)

        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        # return images, relative_poses, triplets_local_indexes, self.triplets_global_indexes[index]
        return images, relative_poses

    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)
    
    def get_total_len(self):
        return super().__len__()
        
    def compute_triplets(self, model):
        self.is_inference = True
        self.compute_triplets_partial(model)
        self.is_inference = False

    def compute_cache(self, model, subset_ds, cache_shape):
        subset_dl = DataLoader(dataset=subset_ds, num_workers=self.compute_triplets_workers,
                               batch_size=self.compute_triplets_batch_size, shuffle=False)
        model = model.eval()
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        with torch.no_grad():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            for images, relative_poses, indexes in tqdm(subset_dl, position=rank, desc=f"compute_cache rank {rank}", ncols=100):
            # for iter, (images, relative_poses, indexes) in enumerate(subset_dl):
                # if iter % 50 == 0 or iter == len(subset_dl) - 1:
                #     print(f'device {model.device} compute_cache, processing batch {iter+1}/{len(subset_dl)}')
                # images = images.view(-1, 3, self.img_shape[0], self.img_shape[1])
                # print('images shape:', images.shape)
                B, S, ch, h, w = images.shape
                relative_poses = relative_poses.view(B, S, 3).to(model.device)
                relative_poses = relative_poses[..., 2] # use only yaw
                
                # if (images.shape[0] % (self.seq_len * self.n_gpus) != 0) and self.n_gpus > 1:
                #     # handle last batch, if it is has less than batch_size sequences
                #     model.module = model.module.to('cuda:1')
                #     # shape[0] is always a multiple of seq_length, sequences are always full size
                #     for sequence in range(images.shape[0] // self.seq_len):
                #         n_seq = sequence * self.seq_len
                #         seq_images = images[n_seq: n_seq + self.seq_len].to('cuda:1')

                #         cache[indexes[sequence], :] = model.module(seq_images).cpu().numpy()

                #     model = model.cuda()
                # else:
                features = model(images.to(model.device), relative_poses.to(model.device))
                cache[indexes.numpy()] = features.cpu().numpy()
        return cache

    def get_best_positive_index(self, qidx, cache, query_features):
        positives_features = cache[self.pIdx[qidx]]
        faiss_index = faiss.IndexFlatL2(self.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.pIdx[qidx][best_positive_num[0]]
        return best_positive_index

    def get_hardest_negatives_indexes(self, cache, query_features, neg_indexes):
        neg_features = cache[neg_indexes]

        faiss_index = faiss.IndexFlatL2(self.features_dim)
        faiss_index.add(neg_features)

        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.nNeg)
        neg_nums = neg_nums.reshape(-1)
        neg_idxs = neg_indexes[neg_nums].astype(np.int32)

        return neg_idxs

    def compute_triplets_partial(self, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        
        sampled_queries_indexes = np.random.choice(self.num_queries, self.cached_queries, replace=False)
        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.num_references, self.cached_negatives, replace=False)
        # print('device:', model.device, 'sampled queries:', sampled_queries_indexes.shape, 'sampled dbs:', sampled_database_indexes.shape)

        positives_indexes = np.unique([idx for db_idx in self.pIdx[sampled_queries_indexes] for idx in db_idx])
        database_indexes = list(sampled_database_indexes) + list(positives_indexes)
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.num_references))
        # print('cache_shape:', (len(self), self.features_dim))
        cache = self.compute_cache(model, subset_ds, cache_shape=(len(self), self.features_dim))
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        for q in tqdm(sampled_queries_indexes, ncols=100, position=rank, desc=f"processing query rank {rank}"):
        # for iter, q in enumerate(sampled_queries_indexes):
            # if iter % 100 == 0 or iter == len(sampled_queries_indexes) - 1:
            #     print(f'device {model.device}, processing query ({iter + 1}/{len(sampled_queries_indexes)})')

            qidx = q + self.num_references
            query_features = cache[qidx]

            best_positive_index = self.get_best_positive_index(q, cache, query_features)
            if isinstance(best_positive_index, np.ndarray):
                best_positive_index = best_positive_index[0]
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.nonNegIdx[q]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            # Take all database images that are negatives and are within the sampled database images
            neg_indexes = self.get_hardest_negatives_indexes(cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((q, best_positive_index, *neg_indexes))

        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)




class TrainingMSLSNoMining_seq(MSLS_seq):
    def __init__(self, root_dir=None, cities=['melbourne'], input_transform=None, task='seq2seq', seq_length=5, subtask='all', exclude_panos=True,
                 nNeg=10, nPos=4):
        self.dataset_cache_dir = 'dataset_cache'
        os.makedirs(self.dataset_cache_dir, exist_ok=True)
        cities_str_md5 = encode_cities_to_md5(cities)
        self.cache_str = os.path.join(self.dataset_cache_dir, f'train_msls_{cities_str_md5}_seqlen_{seq_length}.pth')
        super().__init__(root_dir, cities, input_transform, task, seq_length, subtask, exclude_panos, cache_file=self.cache_str, pos_thresh=10, neg_thresh=25)
        print("Switching to training mode for MSLS_seq online mining dataset...")
        self.nNeg = nNeg                          # Number of negatives to use per query
        self.nPos = nPos                          # Number of positives to use per query
        self.is_inference = False
        self.qIdx = None
        self.pIdx = np.array(self.pIdx, dtype=object)
        # Additional initialization for training mode can be added here



    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)
        # query_index = index + self.num_references
        positive_indexes = self.pIdx[index]
        neg_indexes = []
        all_neg_indexes = np.setdiff1d(np.arange(self.num_references), self.nonNegIdx[index], assume_unique=True)
        if len(all_neg_indexes) >= self.nNeg:
            neg_indexes = np.random.choice(all_neg_indexes, self.nNeg, replace=False)
        else:
            neg_indexes = np.random.choice(all_neg_indexes, self.nNeg, replace=True)
        
        if len(positive_indexes) >= self.nPos:
            positive_indexes = np.random.choice(positive_indexes, self.nPos, replace=False)
        else:
            positive_indexes = np.random.choice(positive_indexes, self.nPos, replace=True)

        query_imgs, query_relative_poses, _ = super().__getitem__(index + self.num_references, shuffle_seq=True)
        positive_imgs = []
        positive_relative_poses = []
        negative_imgs = []
        negative_relative_poses = []
        for pos_idx in positive_indexes:
            pos_imgs, pos_relative_poses, _ = super().__getitem__(pos_idx, shuffle_seq=True)
            positive_imgs.append(pos_imgs)
            positive_relative_poses.append(pos_relative_poses)
        for neg_idx in neg_indexes:
            neg_imgs, neg_relative_poses, _ = super().__getitem__(neg_idx, shuffle_seq=True)
            negative_imgs.append(neg_imgs)
            negative_relative_poses.append(neg_relative_poses)
        # query = torch.stack(
        #     [self.base_transform(Image.open(join(self.dataset_folder, im))) for im in self.q_paths[query_index].split(',')])

        # positive = torch.stack(
        #     [self.base_transform(Image.open(join(self.dataset_folder, im))) for im in self.db_paths[best_positive_index].split(',')])

        # negatives = [torch.stack([self.base_transform(Image.open(join(self.dataset_folder, im)))for im in self.db_paths[idx].split(',')])
        #                 for idx in neg_indexes]

        images = torch.stack((query_imgs, *positive_imgs, *negative_imgs), 0)
        # print('images shape:', images.shape)
        relative_poses = torch.stack((query_relative_poses, *positive_relative_poses, *negative_relative_poses), 0)

        # triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        # for neg_num in range(len(neg_indexes)):
        #     triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        # return images, relative_poses, triplets_local_indexes, self.triplets_global_indexes[index]
        return images, relative_poses
    
    def __len__(self):
        if self.is_inference:
            return super().__len__()
        else:
            return self.num_queries