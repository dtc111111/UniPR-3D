import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from os.path import join
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from dataloaders.GSVCitiesDataset import calculate_relative_transform_geodetic
import torch
from sklearn.neighbors import NearestNeighbors
from dataloaders.CacheDataset import Mat_Redis_Utils

default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam","helsinki",
              "tokyo","toronto","saopaulo","moscow","zurich","paris","bangkok",
              "budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"],
    'val': ["cph", "sf"],
    'test': ["miami","athens","buenosaires","stockholm","bengaluru","kampala"]
}
# posDistThr = 25  # Positive sample distance threshold (meters)

# Input transform (resize to your model input, e.g., 392x518)
def input_transform(image_size=(392, 518)):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MSLS_seq(Dataset):
    def __init__(self, root_dir=None, cities='', input_transform=None, task='seq2seq', seq_length=5, subtask='all', exclude_panos=True, cache_file=None,
                 pos_thresh=25, neg_thresh=25, redis=False):
        super().__init__()
        print("Initializing MSLS_seq dataset...")
        # Ensure val mode
        if root_dir is None:
            root_dir = '/nas0/dataset/mapillary/'
        self.root_dir = root_dir
        self.transform = input_transform
        self.cities = default_cities['val'] if cities == '' else cities
        self.task = task
        self.subtask = subtask
        self.seq_length = seq_length  # Sequence length, must be odd
        self.exclude_panos = exclude_panos
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.use_redis = redis
        if redis:
            self.redis_handle = Mat_Redis_Utils()

        # Assert seq2seq
        assert task == 'seq2seq', "Only seq2seq task is supported"
        assert seq_length % 2 == 1 and seq_length > 1, "seq_length must be odd >1"

        self.qImages = []  # Query sequence paths (comma-separated)
        self.dbImages = []  # Database sequence paths (comma-separated)
        self.qRelativePoses = []  # Relative poses for query sequences
        self.dbRelativePoses = []  # Relative poses for database sequences
        self.qIdx = []     # List of query indices (for positives)
        self.pIdx = []     # List of positive sample indices (for each query, db indices)
        self.nonNegIdx = []  # List of non-negative sample indices (for each query, db indices)

        cache_loaded = False
        if cache_file is not None and os.path.isfile(cache_file):
            print(f'Loading cached data from {cache_file}...')
            cache_dict = torch.load(cache_file, weights_only=False)
            assert cache_dict['cities'] == self.cities, "Cached cities do not match"
            assert cache_dict['seq_length'] == self.seq_length, "Cached seq_length does not match"
            assert cache_dict['pos_thresh'] == self.pos_thresh, "Cached pos_thresh does not match"
            assert cache_dict['neg_thresh'] == self.neg_thresh, "Cached neg_thresh does not match"
            self.__dict__.update(cache_dict)
            cache_loaded = True

        # Load data (similar to msls.py __init__ for val)
        if not cache_loaded:
            for city in self.cities:
                print(f"=====> Loading data for {city}")
                subdir = 'train_val'

                # get len of images from cities so far for indexing
                _lenQ = len(self.qImages)
                _lenDb = len(self.dbImages)

                # Load query and database data
                qData = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col=0)
                qDataRaw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col=0)
                dbData = pd.read_csv(join(root_dir, subdir, city, 'database', 'postprocessed.csv'), index_col=0)
                dbDataRaw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col=0)

                # Arrange as sequences (seq_length_q = seq_length_db = seq_length)
                qSeqKeys, qSeqIdxs = self.arange_as_seq(qData, join(root_dir, subdir, city, 'query'), seq_length)
                dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbData, join(root_dir, subdir, city, 'database'), seq_length)
                # print('qSeqKeys example:', qSeqKeys[0])
                print('qSeqIdxs:', qSeqIdxs.shape, 'dbSeqIdxs:', dbSeqIdxs.shape)
                # Filter based on subtask (as in msls.py)
                qIdx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                dbIdx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                val_frames_q = np.where(qIdx[self.subtask])[0]
                qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, val_frames_q)

                val_frames_db = np.where(dbIdx[self.subtask])[0]
                dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, val_frames_db)

                # filter based on panorama data
                if self.exclude_panos:
                    panos_frames = np.where((qDataRaw['pano'] == False).values)[0]
                    qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, panos_frames)

                    panos_frames = np.where((dbDataRaw['pano'] == False).values)[0]
                    dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, panos_frames)

                print('qSeqIdxs:', qSeqIdxs.shape, 'dbSeqIdxs:', dbSeqIdxs.shape)
                # Extend image lists
                self.dbImages.extend(dbSeqKeys)
                self.qImages.extend(qSeqKeys)
                # Relative poses
                qRelativePosesCity = self.calculate_seq_relative_poses(qSeqIdxs, qDataRaw)
                dbRelativePosesCity = self.calculate_seq_relative_poses(dbSeqIdxs, dbDataRaw)
                self.qRelativePoses.extend(qRelativePosesCity)
                self.dbRelativePoses.extend(dbRelativePosesCity)

                # Unique indices
                unique_qSeqIdx = np.unique(qSeqIdxs)
                unique_dbSeqIdx = np.unique(dbSeqIdxs)

                if len(unique_qSeqIdx) == 0 or len(unique_dbSeqIdx) == 0:
                    continue

                qData = qData.loc[unique_qSeqIdx]
                dbData = dbData.loc[unique_dbSeqIdx]

                # UTM coordinates
                utmQ = qData[['easting', 'northing']].values.reshape(-1, 2)
                utmDb = dbData[['easting', 'northing']].values.reshape(-1, 2)

                # Find positives (similar to msls.py)
                neigh = NearestNeighbors(algorithm='brute')
                neigh.fit(utmDb)
                _, I = neigh.radius_neighbors(utmQ, self.pos_thresh)
                _, I_nonneg = neigh.radius_neighbors(utmQ, self.neg_thresh)

                # Convert to sequence indices (simplified, assume center frame represents sequence)
                seqIdx2frameIdx = lambda seqIdx, seqIdxs : seqIdxs[seqIdx]
                frameIdx2uniqFrameIdx = lambda frameIdx, uniqFrameIdx : np.where(np.in1d(uniqFrameIdx, frameIdx))[0]
                uniqFrameIdx2seqIdx = lambda frameIdxs, seqIdxs : np.where(np.in1d(seqIdxs,frameIdxs).reshape(seqIdxs.shape))[0]
                for q_seq_idx in range(len(qSeqKeys)):
                    q_frame_idxs = seqIdx2frameIdx(q_seq_idx, qSeqIdxs)
                    q_uniq_frame_idx = frameIdx2uniqFrameIdx(q_frame_idxs, unique_qSeqIdx)
                    p_uniq_frame_idxs = np.unique([p for pos in I[q_uniq_frame_idx] for p in pos])
                    nonneg_uniq_frame_idxs = np.unique([p for pos in I_nonneg[q_uniq_frame_idx] for p in pos])

                    # the query image has at least one positive
                    if len(p_uniq_frame_idxs) > 0:
                        p_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[p_uniq_frame_idxs], dbSeqIdxs))

                        self.pIdx.append(p_seq_idx + _lenDb)
                        self.qIdx.append(q_seq_idx + _lenQ)
                        nonneg_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[nonneg_uniq_frame_idxs], dbSeqIdxs))
                        self.nonNegIdx.append(nonneg_seq_idx + _lenDb)


                    # else:
                    #     query_key = qSeqKeys[q_seq_idx].split('/')[-1][:-4]
                    #     self.query_keys_with_no_match.append(query_key)

                # self.qIdx.extend(range(len(self.dbImages), len(self.dbImages) + len(qSeqKeys)))  # Query indices start after db

        print(f'Total queries with positives: {len(self.pIdx)}')
        if self.neg_thresh == self.pos_thresh:
            # nonNegIdx must be same as pIdx
            assert all([set(p) == set(nn) for p, nn in zip(self.pIdx, self.nonNegIdx)]), "nonNegIdx must be same as pIdx when neg_thresh == pos_thresh"
        # Full image list: db + queries
        self.images = self.dbImages + [self.qImages[i] for i in self.qIdx]
        self.relative_poses = self.dbRelativePoses + [self.qRelativePoses[i] for i in self.qIdx]
        # self.relative_poses = np.concatenate((self.dbRelativePoses, np.array(self.qRelativePoses)[self.qIdx]))
        print(len(self.images), "seqs loaded for MSLS Seq dataset.")
        self.num_references = len(self.dbImages)  # Used for val end
        self.ground_truth = self.pIdx  # Used for val end
        self.num_queries = len(self.qIdx)

        if len(self.images) == 0:
            raise ValueError("No images loaded for MSLS Seq")
        
        if cache_file is not None and not cache_loaded:
            cache_dict = {
                'cities': self.cities,
                'seq_length': self.seq_length,
                'qImages': self.qImages,
                'dbImages': self.dbImages,
                'qRelativePoses': self.qRelativePoses,
                'dbRelativePoses': self.dbRelativePoses,
                'qIdx': self.qIdx,
                'pIdx': self.pIdx,
                'nonNegIdx': self.nonNegIdx,
                'pos_thresh': self.pos_thresh,
                'neg_thresh': self.neg_thresh,
            }
            print(f'Saving cached data to {cache_file}...')
            torch.save(cache_dict, cache_file)

    def calculate_seq_relative_poses(self, seqIdxs, DataRaw):
        relative_poses = []
        for seq_idx in seqIdxs:
            pose_center = DataRaw.iloc[seq_idx[len(seq_idx)//2]][['lon', 'lat', 'ca']]
            relative_poses_seq = []
            for frame_idx in seq_idx:
                pose_support = DataRaw.iloc[frame_idx][['lon', 'lat', 'ca']]
                # relative_pose = calculate_relative_transform_geodetic(pose_center, pose_support)
                relative_pose = calculate_relative_transform_geodetic(pose_support, pose_center)
                relative_poses_seq.append(relative_pose)
            relative_poses.append(np.array(relative_poses_seq))
        return relative_poses

    # Copied from msls.py: arrange as sequences
    def arange_as_seq(self, data, path, seq_length):
        seqInfo = pd.read_csv(join(path, 'seq_info.csv'), index_col=0)

        seq_keys, seq_idxs = [], []
        for idx in data.index:
            if idx < (seq_length // 2) or idx >= (len(seqInfo) - seq_length // 2):
                continue
            seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
            seq = seqInfo.iloc[seq_idx]
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])
                seq_keys.append(seq_key)
                seq_idxs.append(seq_idx)
        return seq_keys, np.asarray(seq_idxs)

    # Copied from msls.py: filter
    def filter(self, seqKeys, seqIdxs, center_frame_condition):
        keys, idxs = [], []
        for key, idx in zip(seqKeys, seqIdxs):
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index, center_first=True, shuffle_seq=False):
        paths = self.images[index].split(',')  # Comma-separated sequence paths
        relative_poses = self.relative_poses[index]
        
        if center_first:
            # move center frame to first
            half_len = len(relative_poses) // 2
            paths = paths[half_len:half_len+1] + paths[:half_len] + paths[half_len+1:] # center frame first
            relative_poses = np.concatenate((relative_poses[half_len:half_len+1], relative_poses[:half_len], relative_poses[half_len+1:]), axis=0)
        relative_poses = torch.tensor(relative_poses, dtype=torch.float32)  # (S + 1, 3)

        if shuffle_seq:
            # random shuffle imgs and relative poses from 1
            rand_idx = torch.randperm(len(paths) - 1) + 1  # shuffle from 1 to S
            paths = [paths[0]] + [paths[i] for i in rand_idx]
            relative_poses = torch.cat((relative_poses[0:1], relative_poses[rand_idx]), dim=0)

        seq_images = []
        for p in paths:
            if self.use_redis:
                img = self.redis_handle.load_PIL(p)
            else:
                img = Image.open(p)
            # make sure img is (480, 640)
            # assert img.size == (640, 480), f"Image size is {img.size}, expected (640, 480)"
            if self.transform:
                img = self.transform(img)
            else:
                img = F.to_tensor(img)
            seq_images.append(img)
        img_tensor = torch.stack(seq_images)  # (S + 1, C, H, W)
        return img_tensor, relative_poses, index

    # def getPositives(self):
    #     # Return pIdx: for each query, list of positive db indices (from 0 to num_references-1)
    #     return self.pIdx  # Already computed



if __name__ == "__main__":
    # Example usage
    dataset = MSLS_seq(root_dir='/nas0/dataset/mapillary/', seq_length=5, cache_file='test_cache.pth')
    res = dataset.__getitem__(30)
    print("Image tensor shape:", res[0].shape)
    print("Relative poses:", res[1])
    print("Index:", res[2])