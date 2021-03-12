import numpy as np
import torch
import torch.nn.functional as F
import os
import re
import pandas as pd
from collections import OrderedDict

from torch.utils.data import Dataset
from utils.images import normalize_images
from skimage.io import imread
from ast import literal_eval
from sklearn.model_selection import train_test_split
# from joblib import Parallel, delayed

from tqdm import tqdm

import h5py as h5


class CellDataset(Dataset):
    def __init__(self, splitter, dset_type='train'):
        # type: ['train', 'dev', 'test','all']
        
        assert dset_type in ['train', 'dev', 'test', 'all']

        self.splitter = splitter
        self.dset_type = dset_type

    def __len__(self):
        return len(self.splitter._indices[self.dset_type])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.splitter.get_by_idx(self.splitter._indices[self.dset_type][idx])
        sample = sample.astype("float32")
        sample = sample / 32767.0  # Assume int16
        sample = torch.from_numpy(sample)
        sample = sample.expand((1, -1, -1))
        return sample


class HDF5TrainTestSplitter:
    def __init__(self):
        self.metadata = pd.DataFrame()
        self.fnames = []
        self._data = OrderedDict()
        self._indices = {}
    
    def add_dataset(self, name, fname, metadata, plate_ids=[]):
        self._data[name] = {}
        self._data[name]['fname'] = fname
        self._data[name]['metadata'] = metadata
        metadata.loc[:, 'dataset_name'] = name
        # Add plate ID
        plateid_rexp = re.compile('.*plate-(\d+)_well-.*')
        metadata['plate_id'] = metadata['file_name'].apply(lambda x: plateid_rexp.split(x)[1])
        # Filter by plate ids
        if len(plate_ids) > 0:
            metadata = metadata.loc[metadata['plate_id'].isin(plate_ids), :]
        self.metadata = self.metadata.append(metadata)
        # self.metadata.drop_duplicates(subset=['dataset_name', 'cell_idx'], keep='first', inplace=True)
        self.metadata.reset_index(inplace=True, drop=True)
        self.fnames.append(fname)
    
    def get_by_idx(self, idx):
        dataset_name = self.metadata.loc[idx, "dataset_name"]
        cell_idx = int(self.metadata.loc[idx, "cell_idx"])
        channel_idx = int(self.metadata.loc[idx, "channel_idx"])
        with h5.File(self._data[dataset_name]['fname'], 'r') as f:
            dset = f['nuclei']
            return dset[cell_idx, channel_idx, :, :]

    def train_dev_test_split(self):
        x_train, x_temp, _, y_temp = train_test_split(
            self.metadata[['cell_idx', 'channel_idx', 'dataset_name']],
            self.metadata[['cell_idx', 'channel_idx', 'dataset_name']],
            stratify=self.metadata['dataset_name'],
            test_size=0.3,
            random_state=666,
        )
        x_dev, x_test, _, _ = train_test_split(
            x_temp,
            y_temp,
            stratify=y_temp['dataset_name'],
            test_size=0.5,
            random_state=666,
        )
        self._indices['train'] = x_train.index.to_list()
        self._indices['dev'] = x_dev.index.to_list()
        self._indices['test'] = x_test.index.to_list()
    
    def get_indices_for_data_subset(self, plate_ids=[]):
        plate_idz = [str(pid) for pid in plate_ids]
        if len(plate_ids) == 0:
            self._indices['all'] = self.metadata.index.values
        else:
            self._indices['all'] = self.metadata[self.metadata['plate_id'].isin(plate_idz)].index.values


class NucleiExtractor:
    def __init__(self, imgs_list, cords_df, random_state=1):
        self.imgs_list = imgs_list
        cords_df["bbox"] = [literal_eval(x) for x in cords_df["bbox"]]
        self.cords_df = cords_df
        self.random_state = random_state
        self.croped_cells_df = pd.DataFrame()

    def _extract_cells(self, img, cords_df, bbox_enlargement=0):
        nuclei_list = []
        shape_max_threshold = 65
        shape_min_threshold = 15

        for _, row in cords_df.iterrows():
            minr, minc, maxr, maxc = row.bbox
            patched_nuclei = img[
                minr - bbox_enlargement: maxr + bbox_enlargement,
                minc - bbox_enlargement: maxc + bbox_enlargement,
            ]
            r_shape, c_shape = patched_nuclei.shape
            if (r_shape > shape_min_threshold and r_shape < shape_max_threshold) and \
               (c_shape > shape_min_threshold and c_shape < shape_max_threshold):
                nuclei_list.append(patched_nuclei)
                self.croped_cells_df = self.croped_cells_df.append(row)

        return nuclei_list

    def _extract_single_file(self, img_path, img_idx, bbox_enlargement=0):
        if img_idx % 100 == 0:
            print(f"Processing image {img_idx} ({img_path})...")
        base_name = os.path.basename(img_path)
        idx = base_name.index("_channel")
        base_name = base_name[0:idx]
        # Centroids found on a given image
        nuclei_df = self.cords_df[self.cords_df.file_name == base_name]
        nuclei_df['file_name'] = img_path
        nuclei_df['base_file_name'] = base_name
        img = imread(img_path, plugin="tifffile")
        img = normalize_images(img)
        img_nuclei = self._extract_cells(img, nuclei_df, bbox_enlargement)
        return img_nuclei

    def _extract_all_cells(self, imgs_list, bbox_enlargement=0):
        nuclei_list = []
        print("Extracting individual cells...")
        for img_idx, img_path in enumerate(tqdm(imgs_list)):
            nuclei_list.extend(
                self._extract_single_file(img_path, img_idx, bbox_enlargement)
            )

        return np.array(nuclei_list)
        # nuclei_list = Parallel(n_jobs=8)(
        #    delayed(self._extract_single_file)(img_path, img_idx, bbox_enlargement)
        #    for img_idx, img_path in enumerate(imgs_list)
        # )
        # return [i for j in nuclei_list for i in j]

    def split_and_extract_datasets(self, bbox_enlargement=0):
        # Stratify by channel TODO add other props to sample like nuclei size, pix intensity
        imgs_channel = [x[x.index("_channel"):] for x in self.imgs_list]
        x_train, x_temp, _, y_temp = train_test_split(
            self.imgs_list,
            imgs_channel,
            stratify=imgs_channel,
            test_size=0.3,
            random_state=self.random_state,
        )
        x_dev, x_test, _, _ = train_test_split(
            x_temp,
            y_temp,
            stratify=y_temp,
            test_size=0.5,
            random_state=self.random_state,
        )
        train_set = self._extract_all_cells(x_train, bbox_enlargement)
        dev_set = self._extract_all_cells(x_dev, bbox_enlargement)
        test_set = self._extract_all_cells(x_test, bbox_enlargement)
        return train_set, dev_set, test_set

    def extract_all_cells(self):
        return self._extract_all_cells(self.imgs_list)


def train_dev_test_split(df_metadata: pd.DataFrame):
    # channel_re = re.compile(r'.*channel-(\w+)\.tif')
    # imgs_channel = df_metadata['file_name'].apply(lambda x: channel_re.split(x)[1])
    x_train, x_temp, _, y_temp = train_test_split(
        df_metadata[['Column1', 'channel', 'base_file_name']],
        df_metadata['channel'],
        stratify=df_metadata['channel'],
        test_size=0.3,
        random_state=666,
    )
    x_dev, x_test, _, _ = train_test_split(
        x_temp,
        y_temp,
        stratify=y_temp,
        test_size=0.5,
        random_state=666,
    )
    return x_train, x_dev, x_test
