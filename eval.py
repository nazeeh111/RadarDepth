import open3d as o3d
import argparse
import os, sys
import torch
import rerun as rr
from typing import List
import numpy as np

from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
from dataloader.dataset import Dataset
from utils.pairing import radarimg2lidar

def load_raw_lidar(path_to_lidar: str, T_rl: np.ndarray) -> List[np.ndarray]:
    ret_val = []
    for i, filename in enumerate(sorted(os.listdir(path_to_lidar))):
        if filename.endswith(".npy"):
            path_to_file = os.path.join(path_to_lidar, filename)
            lidar_points = np.load(path_to_file, allow_pickle=True).astype(np.float32)[:, :3]

            pts_in_radar_frame = (T_rl[:3, :3] @ lidar_points.T + T_rl[:3, 3:]).T

            ret_val.append(pts_in_radar_frame)
    return ret_val

def process_dataset(
        model: DRNet3D,
        dataset: Dataset,
        radar_calib_npz: str,
        corresponding_lidar_pts: List[np.ndarray]
    ):

    sm = torch.nn.Softmax(dim=1)
    th = 0.4

    rr.init("PCD Comparison")
    rr.spawn()

    for index, data in enumerate(dataset):
        data = dataset[index]
        input = data['radar']
        input = torch.from_numpy(input).double().unsqueeze(0).unsqueeze(0)
        output_dict = model(input)
        output, mask = output_dict['depth'], output_dict['segment_mask']
        output = output.squeeze(0).cpu().detach().numpy()
        mask = sm(mask).squeeze(0).cpu().detach().numpy().copy()
        output[mask[0, :, :] < th] = 0
        radar_unprojected = radarimg2lidar(radar_calib_npz, output, True)

        lidar_img = data['lidar']
        lidar_mask = data['lidar_mask']
        lidar_img[lidar_mask[0, :, :] == 0] = 0
        lidar_pts = radarimg2lidar(radar_calib_npz, lidar_img)

        rr.set_time("frame", sequence=index)
        rr.set_time("timestamp", timestamp=index / 10) # This is a rough estimate, radar data is roughly 10Hz.
        rr.log("world/radar", rr.Points3D(
            positions=radar_unprojected,
            colors=[255, 0, 0],  # red
            radii=0.02
        ))

        rr.log("world/label_lidar", rr.Points3D(
            positions=lidar_pts,
            colors=[0, 255, 0],  # green
            radii=0.02
        ))

        rr.log("world/raw_lidar", rr.Points3D(
            positions=corresponding_lidar_pts[index],
            colors=[0, 0, 255],  # blue
            radii=0.02
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('-m', '--model_path', type=str, help='Path to the model file.', required=True)
    parser.add_argument('-c', '--radar_calib_npz', type=str, help='directory to radar mapping config', required=True)
    parser.add_argument('-r', '--rig_calibration_npz', type=str, help='extrinsic calibration of the rig', required=True)
    parser.add_argument('-d', '--dataset_dir', type=str, help='directory of dataset in training dataset format', required=True)
    parser.add_argument('-l', '--lidar_dir', type=str, help='directory of raw lidar pts', required=True)
    args = parser.parse_args()

    model_path = args.model_path
    dataset_dir = args.dataset_dir
    lidar_dir = args.lidar_dir
    radar_calib_npz = args.radar_calib_npz

    model = DRNet3D(DEPTH_VAL)
    model.load_state_dict(torch.load(model_path))
    model.double()
    model.eval()

    dataset = Dataset(dataset_dir)

    rig_calib = np.load(args.rig_calibration_npz)
    T_bl = rig_calib['T_bl']
    T_br = rig_calib['T_br']
    T_rl = np.linalg.inv(T_br) @ T_bl
    lidar_pts = load_raw_lidar(lidar_dir, T_rl)
    process_dataset(model, dataset, radar_calib_npz, lidar_pts)
