# This is needed since apparently the vanilla dropped some frames for radar, in order to be kept consistent, I'm adding this back in
from typing import Tuple
from .ColoRadar_tools import dataset_loaders as dsl
from .ColoRadar_tools import plot_pointclouds as ppc
from scipy.spatial.transform import Rotation
import numpy as np
import argparse

class CascadeDataLoader:
    def __init__(self, args):

        all_radar_param = ppc.get_cascade_params(args.calib)
        self.radar_params = all_radar_param["heatmap"]
        self.radar_params['T_bs'] = np.eye(4)
        self.radar_params['T_bs'][:3,3] = self.radar_params['translation']
        self.radar_params['T_bs'][:3,:3] = Rotation.from_quat(self.radar_params['rotation']).as_matrix()
        # self.radar_pc_precalc = ppc.get_heatmap_points(self.radar_params, args=args)
        # self.radar_pc_polar_precal = self.getRTPCoordinates()

        self.lidar_params = dsl.get_lidar_params(args.calib)
        self.lidar_params['T_bs'] = np.eye(4)
        self.lidar_params['T_bs'][:3,3] = self.lidar_params['translation']
        self.lidar_params['T_bs'][:3,:3] = Rotation.from_quat(self.lidar_params['rotation']).as_matrix()
        self.lidar_label = self.lidar_params['sensor_type'] + ' ' + self.lidar_params['data_type']

        self.gt_params = dsl.get_groundtruth_params()
        self.gt_label = self.gt_params['sensor_type'] + ' ' + self.gt_params['data_type']

        # IMU
        self.imu_data = dsl.get_imu(args.seq)
        self.imu_param = dsl.get_imu_params(args.calib)

        # Timestamp 
        self.radar_timestamps = dsl.get_timestamps(args.seq, self.radar_params)
        self.gt_timestamps = dsl.get_timestamps(args.seq, self.gt_params)
        self.imu_timestamp = dsl.get_timestamps(args.seq, self.imu_param)
        self.lidar_timestamps = dsl.get_timestamps(args.seq, self.lidar_params)

        # GT Poses
        self.gt_poses = dsl.get_groundtruth(args.seq)
        self.radar_gt, radar_indices = ppc.interpolate_poses(self.gt_poses, 
                                                    self.gt_timestamps, 
                                                    self.radar_timestamps)
        radar_idx = 0
        self.plot_data = []
        while radar_idx < len(radar_indices):
            self.plot_data.append((self.radar_timestamps[radar_indices[radar_idx]],
                            self.radar_gt[radar_idx],
                            radar_indices[radar_idx],
                            'radar'))
            radar_idx += 1

        
        self.lidar_gt, lidar_indices = ppc.interpolate_poses(self.gt_poses, 
                                                    self.gt_timestamps, 
                                                    self.lidar_timestamps)
        self.lidar_plot_data = []
        lidar_idx = 0
        while lidar_idx < len(lidar_indices):
            self.lidar_plot_data.append((self.lidar_timestamps[lidar_indices[lidar_idx]],
                            self.lidar_gt[lidar_idx],
                            lidar_indices[lidar_idx],
                            'lidar'))
            lidar_idx += 1
        self.args = args

        if self.args.seq[-1] != "/":
            self.args.seq = self.args.seq + "/"

    def lidarPoints(self, idx):
        lidar_pc_local = dsl.get_pointcloud(self.lidar_plot_data[idx][2], self.args.seq, self.lidar_params)
        # downsample for faster plotting
        # lidar_pc_local = ppc.downsample_pointcloud(lidar_pc_local, 0.1)
        return lidar_pc_local, self.lidar_plot_data[idx][1], self.lidar_plot_data[idx][0]
        
        # # transform pointcloud to plotting frame
        # T_ws = np.dot(R_wb, self.lidar_params['T_bs'])
        # lidar_pc = ppc.transform_pcl(lidar_pc_local, T_ws)
        # lidar_pc = ppc.remove_ceiling_floor(lidar_pc)

    # def __getitem__(self, idx):
    #     # Sequence 1:
    #     radar_hm = dsl.get_heatmap(self.plot_data[idx][2], self.args.seq, self.radar_params)
    #     temp_precal = self.radar_pc_precalc
    #     temp_precal[:,3:5] = radar_hm[:,:,self.args.min_range:,:].reshape(-1,2)
    #     radar_pc_local = ppc.downsample_pointcloud(temp_precal, self.args.voxel_size)
    #     radar_pc_local = numericalUtils.normalizePC(radar_pc_local, 5e4, 'tanh')
    #     radar_pc_local = radar_pc_local[radar_pc_local[:,3] > self.args.threshold]
    #     return radar_pc_local

    def radarHM(self, idx) -> Tuple[np.ndarray, np.ndarray, float]:
        radar_hm = dsl.get_heatmap(self.plot_data[idx][2], self.args.seq, self.radar_params)
        # print(f"Associated ts: {self.plot_data[idx][0]}")
        # temp_precal = self.radar_pc_polar_precal
        # temp_precal[:, 3:5] = radar_hm[:,:,self.args.min_range:,:].reshape(-1, 2)
        # radar_pc_local = numericalUtils.normalizePC(temp_precal, 5e4, 'tanh')
        # radar_pc_local = radar_pc_local[radar_pc_local[:,3] > args.threshold]
        return radar_hm, self.plot_data[idx][1], self.plot_data[idx][0]

    def radarPose(self, idx):
        return self.plot_data[idx][1]

    def radarTs(self, idx):
        return self.plot_data[idx][0]
    
    def getRTPCoordinates(self):
        r = np.arange(self.args.min_range, self.radar_params["num_range_bins"])
        az = np.arange(self.radar_params["num_azimuth_bins"])
        el = np.arange(self.radar_params["num_elevation_bins"])
        # ee, yy, zz = np.meshgrid(el, az, r)
        rr, aa, ee = np.meshgrid(r, az, el)
        rae = np.hstack((rr.reshape(-1, 1), aa.reshape(-1, 1), ee.reshape(-1, 1)))
        (r, c) = rae.shape
        rae = np.hstack((rae, np.zeros((r, 2))))
        return rae

    def getGTPoses(self, idx = None):
        # if idx == None, return all
        # {'position': [x,y,z], 'orientation': [x,y,z,w]}
        if idx is None:
            return self.gt_poses
        
        return self.gt_poses[idx]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example DBSCAN")
    parser.add_argument('-s', '--seq', type=str, help='directory of the sequence')
    parser.add_argument('-c', '--calib', type=str, help='directory of calib data')
    parser.add_argument('-t', '--threshold', type=float, default=0.2, help='intensity threshold for plotting heatmap points')
    parser.add_argument('-mr', '--min_range', type=int, default=10, help='if plotting heatmaps, minimum range bin to start plotting')
    parser.add_argument('-hm', '--plot_heatmap', type=str, default='false', help='true to plot radar heatmaps, false to plot pointclouds')
    parser.add_argument('-v', '--voxel_size', type=float, default=0.3, help='Downsample voxel size for faster plotting')
    parser.add_argument('-sc', '--single_chip', type=str, default='true', help='true to plot single chip data, false to plot cascade data')
    parser.add_argument('-p', '--plot', type=str, default='true', help='plot example data')
    args = parser.parse_args()
    args = parser.parse_args()
    args.single_chip = args.single_chip.lower() == 'true'
    args.plot_heatmap = args.plot_heatmap.lower() == 'true'

    crDL = CascadeDataLoader(args)
    # radar_pc_local = crDL[0]