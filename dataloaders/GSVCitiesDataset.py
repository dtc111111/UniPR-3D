# https://github.com/amaralibey/gsv-cities

import pandas as pd
import numpy as np
import math
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from pyproj import Geod
import torchvision.transforms as T
import multiprocessing as mp
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = '/nas0/dataset/gsv-cities/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

def sample_near(df, sample_indices, k, include_self=False, random_state=None):
    """
    为每个采样位置独立返回其附近的k个样本
    
    参数:
    df: pandas DataFrame
    sample_indices: 列表，需要在其附近采样的索引
    k: 整数，每个索引附近采样的数量
    include_self: 布尔值，是否包含原始索引本身
    random_state: 随机种子，用于可重复结果
    
    返回:
    字典, 键为原始索引, 值为附近采样的DataFrame
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    results = {}
    for idx in sample_indices:
        # 获取当前索引在DataFrame中的位置
        pos = df.index.get_loc(idx)
        # print(f'Index: {idx}, Position: {pos}')
        # 计算附近的范围
        start = max(0, pos - k)
        end = min(len(df) - 1, pos + k)
        
        # 创建附近索引的列表
        nearby_indices = list(range(start, end + 1))
        
        # 如果不包含自身，移除当前索引
        if not include_self and pos in nearby_indices:
            nearby_indices.remove(pos)
        
        selected_indices = np.random.choice(nearby_indices, k, replace=True)
        
        # 将位置转换为实际索引
        actual_indices = [df.index[i] for i in selected_indices]
        
        # 获取对应的DataFrame行
        results[idx] = df.loc[actual_indices]
    
    return results


def calculate_relative_transform_geodetic(point1, point2):
    """
    使用大地测量方法计算相对变换
    """
    lat1, lon1, heading1 = point1
    lat2, lon2, heading2 = point2
    
    # 1. 计算两点间的大地线距离和方位角
    geod = Geod(ellps='WGS84')
    # 返回：正向方位角，反向方位角，距离
    az12, az21, distance = geod.inv(lon1, lat1, lon2, lat2)
    
    # 2. 计算相对位置（在point1的局部坐标系中）
    # 从point1到point2的方位角（相对于真北）
    bearing_to_point2 = az12
    # print(f"bearing_to_point2: {bearing_to_point2}")
    
    # 将距离和方位角转换为局部坐标
    # 注意：heading1是point1的朝向，bearing_to_point2是到point2的方向
    relative_angle = bearing_to_point2 - heading1
    # relative_angle = bearing_to_point2
    
    # 角度归一化
    relative_angle = (relative_angle + 360) % 360
    if relative_angle > 180:
        relative_angle -= 360
    
    relative_angle_rad = math.radians(relative_angle)
    
    relative_x = distance * math.sin(relative_angle_rad)  # 右侧为正
    relative_y = distance * math.cos(relative_angle_rad)  # 前方为正
    
    # 3. 计算相对旋转角度
    relative_theta = heading2 - heading1
    relative_theta = (relative_theta + 180) % 360 - 180 # -180到180范围内

    return relative_x, relative_y, math.radians(relative_theta)


class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        # self.support_frames_num = 0

        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')
    
    # def get_support_frames_num(self):
    #     return self.support_frames_num
    
    # def set_support_frames_num(self, n):
    #     self.support_frames_num = n
    
    def __getitem__(self, item):
        index, support_frames_num = item
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place_ori = self.dataframe.loc[place_id]
        place_ori = place_ori.sort_values(
            by=['year', 'month', 'lat'], ascending=False)
        place_ori = place_ori.reset_index(drop=False)  # reset index to default 0,1,2,...
        # print('All places:\n', place_ori)
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place_ori.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place_ori[: self.img_per_place]
            
        sample_indices = place.index.tolist()
        # print('Sample places:\n', place)
        # print('Sample indices:', sample_indices)
        # support_frames_num = self.get_support_frames_num()
        support_frames = sample_near(
            place_ori, sample_indices, support_frames_num, include_self=False)

        img_paths = []
        relative_poses = []
        for idx in sample_indices:
            anchor_row = place_ori.loc[idx]
            anchor_pose = self.get_img_pose(anchor_row)
            # print(f'Anchor row: {anchor_row}')
            img_names = [self.get_img_name(anchor_row)]
            relative_poses.append(np.array([0.0, 0.0, 0.0]))  # anchor relative pose is always (0,0,0)
            for _, supp_row in support_frames[idx].iterrows():
                support_frame_pose = self.get_img_pose(supp_row)
                # relative_pose = calculate_relative_transform_geodetic(anchor_pose, support_frame_pose)
                relative_pose = calculate_relative_transform_geodetic(support_frame_pose, anchor_pose)
                img_names.append(self.get_img_name(supp_row))
                relative_poses.append(relative_pose)
                # print(f'  Support row: {supp_row}')
            for img_name in img_names:
                img_path = self.base_path + 'Images/' + \
                    anchor_row['city_id'] + '/' + img_name
                img_paths.append(img_path)
        
        imgs = []
        for img_path in img_paths:
            img = self.image_loader(img_path)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        relative_poses = np.stack(relative_poses, axis=0).astype(np.float32)  # shape [K*(S+1), 3]
        relative_poses = torch.tensor(relative_poses).view(self.img_per_place, support_frames_num + 1, 3)  # reshape to [K, S+1, 3]
        
        imgs = torch.stack(imgs).view(self.img_per_place, support_frames_num + 1, *imgs[0].shape) # reshape to [K, S+1, C, H, W]
        # NOTE: contrary to image classification where __getitem__ returns only one image
        # in GSVCities, we return a place, which is a Tensor of K * (S + 1) images (K=self.img_per_place, S=self.support_frames_num)
        # this will return a Tensor of shape [K, S + 1, channels, height, width]. This needs to be taken into account
        # in the Dataloader (which will yield batches of shape [BS, K, S + 1, channels, height, width])
        return imgs, relative_poses, torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row['place_id'] % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name
    
    @staticmethod
    def get_img_pose(row):
        # given a row from the dataframe
        # return the corresponding image name

        lat = row['lat']
        lon = row['lon']
        northdeg = row['northdeg']
        # Convert to UTM coordinates

        return np.array([lat, lon, northdeg])

if __name__ == '__main__':
    # for testing purposes
    dataset = GSVCitiesDataset(
        cities=['London', 'Boston'],
        img_per_place=4,
        min_img_per_place=4,
        random_sample_from_each_place=True,
        transform=default_transform
    )
    # dataset.set_support_frames_num(4)
    res = dataset.__getitem__((10, 3))
    print("Image tensor shape:", res[0].shape)
    print("Relative poses shape:", res[1].shape)
    print("Relative poses:", res[1])
    print("place ids:", res[2])
    # print(f'# of places: {len(dataset)}')
    # print(f'# of images: {dataset.total_nb_images}')
    # print(f'Example place shape (K, S+1, C, H, W): {dataset[0][0].shape}')
