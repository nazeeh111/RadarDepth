import argparse
from typing import Tuple
import open3d as o3d
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
RADAR_ELEVATION_RANGE = (8, 24)

def gen_mid_points(num_sample : int) -> Tuple[np.ndarray]:
    '''
    Gives 5, returns 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 
    '''
    standard_sample = np.arange(num_sample)
    mid_pts = standard_sample[:-1] + np.diff(standard_sample) / 2
    mid_pts = np.sort(np.concatenate((standard_sample, mid_pts)))

    return standard_sample, mid_pts

def radarimg2lidar(calib_fname : str, radar_img : np.ndarray, lidar_frame : bool = False) -> np.ndarray:

    calib_data = np.load(calib_fname)
    range_unit = calib_data['range_bin_width']

    (h, w) = radar_img.shape

    azimuth_lut = calib_data['azimuth_bins'].reshape((1, -1))
    elevation_lut = calib_data['elevation_bins'].reshape((-1, 1))[RADAR_ELEVATION_RANGE[0]:RADAR_ELEVATION_RANGE[1]]

    if w != 128:
        int_indice, mid_indice = gen_mid_points(len(azimuth_lut[0, :]))
        azimuth_lut = np.interp(mid_indice, int_indice, azimuth_lut[0, :]).reshape((1, -1))
    
    # assert w == 128
    if h != 16:
        int_indice, mid_indice = gen_mid_points(len(elevation_lut))
        elevation_lut = np.interp(mid_indice, int_indice, elevation_lut[:, 0]).reshape((-1, 1))

    if lidar_frame and (w != 128) and (h != 16):
        elevation_lut = elevation_lut[7:26]
        radar_img = radar_img.copy()[7:26, :]
    
    ele_val, azi_val = np.meshgrid(elevation_lut, azimuth_lut)
    ele_val = ele_val.T.flatten()
    azi_val = azi_val.T.flatten()
    
    r = radar_img.flatten()
    x = r * np.cos(ele_val) * np.cos(azi_val)
    y = r * np.cos(ele_val) * np.sin(azi_val)
    z = r * np.sin(ele_val)

    return np.vstack((x, y, z)).T

def lidar2radar_dense(calib_fname, T_rl, lidar) -> np.ndarray:
    # 1. move all lidar into radar frame
    # 2. Get radar fov
    # 3. Filter out lidar points out side of radar fov

    lidar_in_r = lidar.point['positions'].clone().numpy()
    radius = np.linalg.norm(lidar_in_r, axis=1)
    lidar_in_r = lidar_in_r[radius > 0]
    radius = radius[radius > 0]
    lidar_type = lidar_in_r.dtype
    T_rl = T_rl.astype(lidar_type)

    lidar_in_r = T_rl[:3, :3] @ lidar_in_r.T + T_rl[:3, 3:]
    radius = np.linalg.norm(lidar_in_r, axis=0)
    lidar_azi = np.arctan2(lidar_in_r[1], lidar_in_r[0])
    lidar_ele = np.arcsin(lidar_in_r[2] / radius)

    calib_data = np.load(calib_fname)
    azimuth_bins = np.array(calib_data['azimuth_bins']).astype(lidar_type)
    ele_bins = np.array(calib_data['elevation_bins'])[[RADAR_ELEVATION_RANGE[0], RADAR_ELEVATION_RANGE[1]]].astype(lidar_type)
    min_azi, max_azi = azimuth_bins.min(), azimuth_bins.max()
    min_ele, max_ele = ele_bins.min(), ele_bins.max()

    azi_mask = np.bitwise_and(lidar_azi > min_azi, lidar_azi < max_azi)
    ele_mask = np.bitwise_and(lidar_ele > min_ele, lidar_ele < max_ele)
    
    return lidar_in_r.T[np.bitwise_and(azi_mask, ele_mask)]

def lidar2radar(calib_fname, T_rl, lidar, elev_range=RADAR_ELEVATION_RANGE, **kwargs) -> np.ndarray:
    debug_info = kwargs.pop('debug', False)
    debug_plot = kwargs.pop('plot', False)

    calib_data = np.load(calib_fname)
    range_unit = calib_data['range_bin_width']

    azimuth_lut = np.rad2deg(calib_data['azimuth_bins'].reshape((1, -1)))
    elevation_lut = np.rad2deg(calib_data['elevation_bins'].reshape(
        (-1, 1))[elev_range[0]:elev_range[1]])

    if debug_info:
        print(azimuth_lut)
        print(elevation_lut)

    azimuth_length = len(azimuth_lut[0, :])
    elevation_length = len(elevation_lut)

    lidar_in_r = lidar.point['positions'].clone().numpy()
    radius = np.linalg.norm(lidar_in_r, axis=1)
    lidar_in_r = lidar_in_r[radius > 0]
    radius = radius[radius > 0]

    lidar_in_r = T_rl[:3, :3] @ lidar_in_r.T + T_rl[:3, 3:]
    radius = np.linalg.norm(lidar_in_r, axis=0)
    azimuth = np.rad2deg(np.arctan2(lidar_in_r[1], lidar_in_r[0]))
    elevation = np.rad2deg(np.arcsin(lidar_in_r[2] / radius))

    # No vectorization, stupid enumeration now
    azimuth_lut_expanded = np.tile(azimuth_lut, (len(azimuth), 1))
    azimuth_expanded = np.tile(np.expand_dims(azimuth, axis=1), (1, azimuth_length))

    azimuth_ind = np.argmin(np.abs(azimuth_lut - azimuth_expanded), axis=1)

    elevation_lut_expanded = np.tile(elevation_lut, (1, len(elevation)))
    elevation_expanded = np.tile(np.expand_dims(elevation, axis=0), (elevation_length, 1))

    azimuth_ind = np.argmin(np.abs(azimuth_lut_expanded - azimuth_expanded),
                            axis=1)
    elevation_ind = np.argmin(np.abs(elevation_lut_expanded -
                                     elevation_expanded),
                              axis=0)

    depth_map = np.zeros((len(elevation_lut), azimuth_length))
    depth_map[elevation_ind, azimuth_ind] = radius

    depth_map[np.isclose(depth_map, 0)] = 10

    if debug_plot:
        plt.imshow(depth_map)
        plt.show()

    return depth_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar_npy')
    parser.add_argument('--radar_npz')
    parser.add_argument('--radar_calib_npz')
    parser.add_argument('--rig_calib_npz')
    args = parser.parse_args()

    calib = np.load(args.rig_calib_npz)
    T_bl = o3d.core.Tensor(calib['T_bl'], dtype=o3d.core.Dtype.Float32)
    T_br = o3d.core.Tensor(calib['T_br'], dtype=o3d.core.Dtype.Float32)

    # Lidar to radar
    T_rl = T_br.inv() @ T_bl
    single_slice_ele_range = RADAR_ELEVATION_RANGE
    radar_xyz = load_radar_calib(args.radar_calib_npz, elev_range=single_slice_ele_range)

    radar = load_radar_as_pcd(args.radar_npz, radar_xyz, elev_range=single_slice_ele_range, use_cfar_image=True)
    lidar = load_lidar_as_pcd(args.lidar_npy)

    radar_space_img = lidar2radar(args.radar_calib_npz, T_rl.numpy(), lidar, plot=True, upsample = True)

    lidar_unprojected = radarimg2lidar(args.radar_calib_npz, radar_space_img)
    tpcd = o3d.t.geometry.PointCloud(
        o3d.core.Tensor(lidar_unprojected.astype(np.float32), dtype=o3c.Dtype.Float32))
    
    heat = o3d.core.Tensor(colorize(radar_space_img.flatten()),
                        dtype=o3c.Dtype.Float32)
    tpcd.point['colors'] = heat

    tpcd.transform(T_br)

    # Transform to the body coordinate system
    radar.transform(T_br)
    lidar.transform(T_bl)
    o3d.visualization.draw([tpcd, lidar])
