import argparse
from copy import deepcopy
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import os, sys
sys.path.append("../")

from utils.lidar import load_lidar_as_pcd
from utils.pairing import lidar2radar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Lidar projected to radar dataset")
    parser.add_argument('-l', '--lidar_sequence', type=str, help='directory of lidar sequence')
    parser.add_argument('-r', '--radar_sequence', type=str, help='directory of radar sequence')
    parser.add_argument('-m', '--radar_calib_npz', type=str, help='npz of radar data')
    parser.add_argument('-c', '--rig_calib_npz', type=str, help='npz of rig calib')
    parser.add_argument('-o', '--output', type=str, help="output directory of data")
    parser.add_argument('-i', '--image_gen', type=bool, default=False, help="Generate to image for visualization")
    parser.add_argument('-lower', '--lower_range', type=int, default=11, help='Lower range on radar elevation range')
    parser.add_argument('-upper', '--upper_range', type=int, default=21, help='Upper range on radar elevation range')

    args = parser.parse_args()

    calib = np.load(args.rig_calib_npz)
    T_bl = o3d.core.Tensor(calib['T_bl'], dtype=o3d.core.Dtype.Float32)
    T_br = o3d.core.Tensor(calib['T_br'], dtype=o3d.core.Dtype.Float32)

    # Lidar to radar
    T_rl = T_br.inv() @ T_bl
    elevation_range = (args.lower_range, args.upper_range)

    lidar_path = Path(args.lidar_sequence)
    assert lidar_path.exists() and lidar_path.is_dir(), \
        "Lidar Directory doesn't exists or not a directory"
    radar_path = Path(args.radar_sequence)
    assert radar_path.exists() and radar_path.is_dir(), \
        "Radar Directory doesn't exists or not a directory"

    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    output_path.joinpath('radar').mkdir(exist_ok=True)

    lidar_str = 'lidar'
    output_path.joinpath(lidar_str).mkdir(exist_ok=True)

    if args.image_gen:
        img_output = deepcopy(output_path)
        img_output = output_path.joinpath("img")
        img_output.mkdir(exist_ok=True)

    existing_data_num = 0
    
    for files in tqdm(os.listdir(radar_path)):
        files_pathobj = radar_path.joinpath(files)
        radar = np.load(files_pathobj)
        f_intensity = radar['intensity'][elevation_range[0]:elevation_range[1], :, :]
        f_intensity = np.transpose(f_intensity, (2, 0, 1))
        vel = radar['velocity'][elevation_range[0]:elevation_range[1], :, :]
        vel = np.transpose(vel, (2, 0, 1))

        radar_output_path = output_path.joinpath(f"radar/{existing_data_num + int(files[:-4]):05}")
        np.savez(radar_output_path.resolve().as_posix(),\
            intensity=f_intensity, velocity=vel)

    for files in tqdm(os.listdir(lidar_path)):
        files_pathobj = lidar_path.joinpath(files)
        lidar = load_lidar_as_pcd(files_pathobj)
        lidar_img = lidar2radar(args.radar_calib_npz, T_rl.numpy(), lidar, elev_range=elevation_range)

        lidar_output_path = output_path.joinpath(f"{lidar_str}/{existing_data_num + int(files[:-4]):05}")
        np.save(lidar_output_path.resolve().as_posix(), lidar_img)

        if args.image_gen:
            I8 = (((lidar_img - lidar_img.min()) / (lidar_img.max() - lidar_img.min())) * 255.9).astype(np.uint8)
            img = Image.fromarray(I8)
            img.save(img_output.joinpath(files[:-4] + ".jpg").resolve().as_posix())
