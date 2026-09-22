import numpy as np
import torch
from pathlib import Path
import os
import glob
import argparse


class Dataset(torch.utils.data.Dataset):
    '''
    Generic Dataloader for Radar and/or Lidar data

    Parameters
    ----------
        dataset_path : str
        Path to data directory, input argument takes aboth lidar path and radar path
        so it will be generic enough for swaping between 3d/2d/1d cases
    '''
    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        radar_path = os.path.join(dataset_path, 'radar')
        lidar_str = 'lidar'
        lidar_path = os.path.join(dataset_path, lidar_str)

        assert os.path.isdir(radar_path), f"radar directory must exist {radar_path}"
        assert os.path.isdir(lidar_path), f"lidar directory must exist {lidar_path}"

        self.radar_fnames = sorted(glob.glob(os.path.join(radar_path,
                                                          '*.npz')))
        self.lidar_fnames = sorted(glob.glob(os.path.join(lidar_path,
                                                          '*.npy')))

        assert len(self.radar_fnames) == len(
            self.lidar_fnames), "Lidar and radar file count unmatch"

    def __len__(self):
        return len(self.radar_fnames)

    def __getitem__(self, index):
        radar = np.load(os.path.join(self.radar_fnames[index]))
        lidar = np.load(os.path.join(self.lidar_fnames[index]))

        lidar_true_mask =  np.expand_dims(lidar < 0.0592944 * 128, axis=0)
        lidar_false_mask = np.expand_dims(lidar > 0.0592944 * 128, axis=0)
        lidar_mask = np.concatenate((lidar_true_mask, lidar_false_mask))

        return {
            'radar': radar['intensity'],
            'lidar': lidar,
            'velocity' : radar['velocity'],
            'lidar_mask' : lidar_mask
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             shuffle=True)

    for i, data in enumerate(dataloader):
        print(data['radar'].shape)
        print(data['lidar'].shape)
