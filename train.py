from os import PathLike
from typing import Callable
import argparse
import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau

random_seed = 10

from networks.DepthRefinement.model import DRNet3D, huber_loss, DEPTH_VAL
from dataloader.dataset import Dataset

def prepare_dataset(path : PathLike, bs : int, split : float, shuffle : bool):
    '''
    Return torch type dataset from path with split train and test
    '''
    path_obj = Path(path)
    assert path_obj.exists() and path_obj.is_dir()

    train_datasets = torch.utils.data.ConcatDataset(
        [
            Dataset(path_obj.joinpath(f"run{index}/").resolve().as_posix()) for index in [2]
        ]
    )

    test_datasets = torch.utils.data.ConcatDataset(
        [
            Dataset(path_obj.joinpath(f"run{index}/").resolve().as_posix()) for index in [0, 1, 4]
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        train_datasets, pin_memory=True, batch_size=bs, shuffle=shuffle, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_datasets, pin_memory=True, batch_size=bs, shuffle=shuffle, num_workers=4, drop_last=True)

    return train_loader, test_loader

def loss_fn(output_depth : torch.Tensor, output_mask : torch.Tensor,\
    true_depth : torch.Tensor, true_mask : torch.Tensor):
    '''
    '''
    depth_loss = huber_loss(output_depth, true_depth)

    if output_mask is None:
        return depth_loss

    segment_border_mask = torch.ones_like(true_mask, dtype=torch.bool)
    segment_border_mask[:, :, :2, :] = False
    segment_border_mask[:, :, -2:, :] = False

    segment_loss_fn = nn.BCEWithLogitsLoss()
    segment_loss = segment_loss_fn(output_mask[segment_border_mask], true_mask[segment_border_mask])
    return depth_loss + segment_loss


def train(model : torch.nn.Module, optim : torch.optim.Optimizer, \
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau, \
    train_loader : Data.DataLoader, test_loader : Data.DataLoader,\
    device : torch.device, epoch_num : int, criterion : Callable,\
    output_dir : Path):

    train_loss = np.zeros(epoch_num)
    test_loss = np.zeros(epoch_num)

    for epoch in tqdm(range(epoch_num)):
        this_train_loss = 0
        for i, data in enumerate(train_loader):
            inputs = data['radar'].unsqueeze(1).to(device, dtype=torch.float)
            label = data['lidar'].to(device, dtype=torch.float)
            label_mask = data['lidar_mask'].to(device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_dict = model(inputs)
            output_depth = output_dict['depth']
            output_label = output_dict['segment_mask']

            loss = criterion(output_depth, output_label, label, label_mask)
            this_train_loss = this_train_loss + loss
            loss.backward()
            optim.step()

        last_train_loss = this_train_loss.item() / len(train_loader)
        train_loss[epoch] = last_train_loss
        
        this_total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs = data['radar'].unsqueeze(1).to(device, dtype=torch.float)
                label = data['lidar'].to(device, dtype=torch.float)
                label_mask = data['lidar_mask'].to(device, dtype=torch.float)
                
                output_dict = model(inputs)
                output_depth = output_dict['depth']
                output_label = output_dict['segment_mask']

                loss = criterion(output_depth, output_label, label, label_mask)

                this_total_loss = this_total_loss + loss.cpu().detach().numpy()
            test_loss[epoch] = this_total_loss / len(test_loader)

        scheduler.step(this_total_loss)

        print(f"Epoch: {epoch}/{epoch_num}, Train Loss: {last_train_loss}, \
            Test Loss: {test_loss[epoch]}")

        if epoch % 25 == 0 or (epoch == epoch_num - 1):
            torch.save(model.state_dict(), output_dir.joinpath(f"{epoch}.pth"))

    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss , label="Test Loss")
    plt.legend()
    plt.show()
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('-d', '--dataset_directory', type=str, help='directory of dataset', required=True)
    parser.add_argument('-c', '--cuda', type=bool, default=True, help='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('-o', '--output', type=str, help='model output directory')
    parser.add_argument('-l', '--saved_model', type=str, default="", help="existing model")
    parser.add_argument('-s', '--segmented', type=str, default="true", help="Do segmentation on output", required=True)
    args = parser.parse_args()
    args.segmented = args.segmented.lower() == "true"

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}")

    train_loader, test_loader = prepare_dataset(args.dataset_directory, args.batch_size, 0.6, True)
    
    # Prepare Model and Optimizer
    model = DRNet3D(DEPTH_VAL.to(device), segment=args.segmented)
    if args.saved_model != "":
        print("Continuing training using reference model")
        model.load_state_dict(torch.load(args.saved_model))
    model.float()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    train(model, optimizer, scheduler, train_loader, test_loader, device, args.epoch,\
        loss_fn, output_dir)
