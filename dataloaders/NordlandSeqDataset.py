import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
from itertools import product
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from dataloaders.CacheDataset import Mat_Redis_Utils


class NordlandSeqDataset(data.Dataset):
    def __init__(self, root_dir=None, input_transform=None, seq_len=5, pos_thresh=25, neg_thresh=25, reverse_frames=False, cache_file=None):
        super().__init__()
        if root_dir is None:
            root_dir = '/nas0/dataset/NordlandFiltered/nordland/images/test'
        self.root_dir = root_dir
        self.seq_length = seq_len
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.input_transform = input_transform
        self.redis_handle = Mat_Redis_Utils()

        if not os.path.exists(self.root_dir): raise FileNotFoundError(
            f"Folder {self.root_dir} does not exist")


        self.cache_file = cache_file
        if cache_file is not None and os.path.isfile(cache_file):
            logging.info(f'Loading cached data from {cache_file}...')
            cache_dict = torch.load(cache_file)
            assert cache_dict['seq_length'] == self.seq_length, "Cached seq_length does not match"
            assert cache_dict['pos_thresh'] == self.pos_thresh, "Cached pos_thresh does not match"
            assert cache_dict['neg_thresh'] == self.neg_thresh, "Cached neg_thresh does not match"
            self.__dict__.update(cache_dict)
        else:
            self.init_data()

        if reverse_frames:
            self.db_paths = [",".join(path.split(',')[::-1]) for path in self.db_paths]
        
        assert self.q_without_pos + len(self.qIdx) == len(self.q_paths)
        self.q_paths = [self.q_paths[i] for i in self.qIdx]
        self.images_paths = self.db_paths + self.q_paths
        self.num_references = len(self.db_paths)
        self.num_queries = len(self.qIdx)
        self.ground_truth = self.pIdx


    def init_data(self):
        print("Building NordlandSeqDataset...")

        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.root_dir, "database")
        queries_folder = join(self.root_dir, "queries")

        self.db_paths, all_db_paths, db_idx_frame_to_seq = build_sequences(database_folder, seq_len=self.seq_length, desc='loading database...')
        self.q_paths, all_q_paths, q_idx_frame_to_seq = build_sequences(queries_folder, seq_len=self.seq_length, desc='loading queries...')

        q_unique_idxs = np.unique([idx for seq_frames_idx in q_idx_frame_to_seq for idx in seq_frames_idx])
        db_unique_idxs = np.unique([idx for seq_frames_idx in db_idx_frame_to_seq for idx in seq_frames_idx])

        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in all_db_paths[db_unique_idxs]]).astype(np.float64)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in all_q_paths[q_unique_idxs]]).astype(
            np.float64)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = knn.radius_neighbors(self.queries_utms, radius=self.pos_thresh, return_distance=False)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms, radius=self.neg_thresh, return_distance=False)

        self.qIdx = []
        self.pIdx = []
        self.nonNegIdx = []
        self.q_without_pos = 0
        for q in tqdm(range(len(q_idx_frame_to_seq)), ncols=100, desc='Finding positives and negatives...'):
            q_frame_idxs = q_idx_frame_to_seq[q]
            unique_q_frame_idxs = np.where(np.in1d(q_unique_idxs, q_frame_idxs))

            p_uniq_frame_idxs = np.unique(
                [p for pos in self.hard_positives_per_query[unique_q_frame_idxs] for p in pos])

            if len(p_uniq_frame_idxs) > 0:
                p_seq_idx = np.where(np.in1d(db_idx_frame_to_seq, db_unique_idxs[p_uniq_frame_idxs])
                                     .reshape(db_idx_frame_to_seq.shape))[0]

                self.qIdx.append(q)
                self.pIdx.append(np.unique(p_seq_idx))

                nonNeg_uniq_frame_idxs = np.unique(
                    [p for pos in self.soft_positives_per_query[unique_q_frame_idxs] for p in pos])
                nonNeg_seq_idx = np.where(np.in1d(db_idx_frame_to_seq, db_unique_idxs[nonNeg_uniq_frame_idxs])
                                            .reshape(db_idx_frame_to_seq.shape))[0]
                self.nonNegIdx.append(np.unique(nonNeg_seq_idx))
            else:
                self.q_without_pos += 1

        self.qIdx = np.array(self.qIdx)
        self.pIdx = np.array(self.pIdx, dtype=object)
        if self.cache_file is not None:
            save_dict = {
                'db_paths': self.db_paths,
                'q_paths': self.q_paths,
                'database_utms': self.database_utms,
                'queries_utms': self.queries_utms,
                'hard_positives_per_query': self.hard_positives_per_query,
                'soft_positives_per_query': self.soft_positives_per_query,
                'qIdx': self.qIdx,
                'pIdx': self.pIdx,
                'nonNegIdx': self.nonNegIdx,
                'q_without_pos': self.q_without_pos,
                'seq_length': self.seq_length,
                'pos_thresh': self.pos_thresh,
                'neg_thresh': self.neg_thresh
            }
            torch.save(save_dict, self.cache_file)

    def __getitem__(self, index, center_first=True, shuffle_seq=False):
        # old_index = index
        # if index >= self.database_num:
        #     q_index = index - self.database_num
        #     index = self.qIdx[q_index] + self.database_num

        # img = torch.stack([self.input_transform(Image.open(join(self.root_dir, im))) for im in self.images_paths[index].split(',')])
        paths = self.images_paths[index].split(',')  # Comma-separated sequence paths
        if center_first:
            half_len = self.seq_length // 2
            paths = paths[half_len:half_len+1] + paths[:half_len] + paths[half_len+1:]  # center frame first

        if shuffle_seq:
            # random shuffle imgs and relative poses from 1
            rand_idx = torch.randperm(len(paths) - 1) + 1  # shuffle from 1 to S
            paths = [paths[0]] + [paths[i] for i in rand_idx]

        seq_images = []
        for p in paths:
            # img = Image.open(join(self.root_dir, p))
            img = self.redis_handle.load_PIL(join(self.root_dir, p))
            img = self.input_transform(img)
            seq_images.append(img)
        img_tensor = torch.stack(seq_images)  # (S + 1, C, H, W)

        relative_poses = torch.zeros((self.seq_length, 3))  # Dummy relative poses

        return img_tensor, relative_poses, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return (
            f"< {self.__class__.__name__}, ' #database: {self.num_references}; #queries: {self.num_queries} >")

    def get_positives(self):
        return self.pIdx



# def filter_by_cities(x, cities):
#     for city in cities:
#         if x.find(city) > 0:
#             return True
#     return False


def build_sequences(folder, seq_len=5, desc='loading'):
    # if cities != '':
    #     if not isinstance(cities, list):
    #         cities = [cities]
    # base_path = os.path.dirname(folder)
    paths = []
    all_paths = []
    idx_frame_to_seq = []
    # seqs_folders = [join(base_path, '.')]
    # for seq in tqdm(seqs_folders, ncols=100, desc=desc):
    start_index = len(all_paths)
    # print(glob(join(folder, '*')))
    frame_nums = np.array(list(map(lambda x: int(x.split('@')[7]), sorted(glob(join(folder, '*'))))))
    # print(frame_nums)
    full_seq_paths = sorted(glob(join(folder, '*')))
    seq_paths = np.array([s_p for s_p in full_seq_paths])
    # if cities != '':
    #     sample_path = seq_paths[0]
    #     if not filter_by_cities(sample_path, cities):
    #         continue

    # all_paths += list(seq_paths)
    sorted_idx_frames = np.argsort(frame_nums)
    seq_paths = seq_paths[sorted_idx_frames]
    all_paths += list(seq_paths)
    for idx, frame_num in enumerate(frame_nums):
        if idx < (seq_len // 2) or idx >= (len(frame_nums) - seq_len // 2): continue

        # find surrounding frames in sequence
        seq_idx = np.arange(-seq_len // 2, seq_len // 2) + 1 + idx
        if (np.diff(frame_nums[sorted_idx_frames][seq_idx]) == 1).all():
            paths.append(",".join(seq_paths[seq_idx]))
            idx_frame_to_seq.append(seq_idx + start_index)

    return paths, np.array(all_paths), np.array(idx_frame_to_seq)