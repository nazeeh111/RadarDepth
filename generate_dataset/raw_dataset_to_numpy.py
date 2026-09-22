
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import os, sys
sys.path.append("../")

from utils.pose import se3_logmap
from utils.ColoRadarDataLoader import CascadeDataLoader

def main(dataLoader : CascadeDataLoader, outputPath : str) -> None:
    '''Convert raw dataset to npz and npy format

    Parameters
    ----------
        dataLoader : DataLoader.CascadeDataLoader
            DataLoader Class

        outputPath : output path for the dataset
    '''
    
    # Generate Folders
    datasetRoot = Path(outputPath)
    lidarRawPath = datasetRoot.joinpath('raw', 'lidar')
    radarRawPath = datasetRoot.joinpath('raw', 'radar')
    datasetRoot.mkdir(parents=True, exist_ok=True)
    lidarRawPath.mkdir(parents=True, exist_ok=True)
    radarRawPath.mkdir(parents=True, exist_ok=True)

    lidar2DPath = datasetRoot.joinpath('2D', 'lidar')
    radar2DPath = datasetRoot.joinpath('2D', 'radar')
    lidar2DPath.mkdir(parents=True, exist_ok=True)
    radar2DPath.mkdir(parents=True, exist_ok=True)

    lidarImgPath = datasetRoot.joinpath('depthImg', 'lidar')
    radarImgPath = datasetRoot.joinpath('depthImg', 'radar')
    lidarImgPath.mkdir(parents=True, exist_ok=True)
    radarImgPath.mkdir(parents=True, exist_ok=True)

    calibPath = datasetRoot.joinpath('calib')
    if not calibPath.exists():
        print(f"Calibration folder DNE, creating now at {calibPath.resolve()}")
        calibPath.mkdir(parents=True, exist_ok=True)

        calibrationFile = calibPath.joinpath("calibration")
        mappingFile = calibPath.joinpath("radarMapping")
        
        np.savez(calibrationFile.resolve(),\
            T_bl = dataLoader.lidar_params['T_bs'],\
            T_br = dataLoader.radar_params['T_bs'])
        np.savez(mappingFile.resolve(),\
            range_bin_width = dataLoader.radar_params['range_bin_width'],\
            azimuth_bins = np.array(dataLoader.radar_params['azimuth_bins']),\
                elevation_bins =np.array(dataLoader.radar_params['elevation_bins']))
        print("Calibration folder created")
    
    # Get Paths
    seqLength = len(dataLoader.plot_data)

    base_pose_log = np.zeros((seqLength, 6))
    for idx in tqdm(range(seqLength)):

        # Saving Full 3D Data
        (radar_hm, radarBasePose, radarTs) = dataLoader.radarHM(idx)

        (lidarPts0, lp0, lidarTs0) = dataLoader.lidarPoints((idx + 1) * 2 - 1)
        (lidarPts1, lp1, lidarTs1) = dataLoader.lidarPoints((idx + 1) * 2)
        (lidarPts2, lp2, lidarTs2) = dataLoader.lidarPoints((idx + 1) * 2 + 1)
        lidarPts = [lidarPts0, lidarPts1, lidarPts2]

        lidarTs = np.array([lidarTs0, lidarTs1, lidarTs2])
        lidarIdx = np.argmin(np.abs(lidarTs - radarTs))

        np.savez(f"{radarRawPath.resolve().as_posix()}/{idx:03d}", intensity = radar_hm[:, :, :, 0], velocity = radar_hm[:, :, :, 1])
        np.save(f"{lidarRawPath.resolve().as_posix()}/{idx:03d}", lidarPts[lidarIdx])

        base_pose_log[idx, :] = se3_logmap(radarBasePose)
        
    np.savetxt(datasetRoot.joinpath("base_pose"), base_pose_log)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example DBSCAN")
    parser.add_argument('-s', '--seq', type=str, help='directory of the sequence')
    parser.add_argument('-c', '--calib', type=str, help='directory of calib data')
    parser.add_argument('-o', '--outputPath', type=str, default='', help='Output path for dataset generation')
    args = parser.parse_args()
    dataLoader = CascadeDataLoader(args)
    main(dataLoader, args.outputPath)
