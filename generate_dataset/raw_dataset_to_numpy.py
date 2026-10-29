
import argparse
import os
import numpy as np
from pathlib import Path

import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(dataLoader, outputPath: str, max_frames=None) -> None:
    '''Convert raw dataset to npz and npy format

    Parameters
    ----------
        dataLoader : DataLoader.CascadeDataLoader
            DataLoader Class

        outputPath : output path for the dataset
    '''
    
    from utils.pose import se3_logmap

    # Generate Folders
    datasetRoot = Path(outputPath)
    if datasetRoot.exists() and any(datasetRoot.iterdir()):
        raise ValueError('Output must be empty; existing data will not be overwritten')
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
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError('max_frames must be positive')
        seqLength = min(seqLength, max_frames)
    lidar_timestamps = np.asarray([row[0] for row in dataLoader.lidar_plot_data])
    if (not len(lidar_timestamps) or not np.isfinite(lidar_timestamps).all()
            or not np.all(np.diff(lidar_timestamps) >= 0)):
        raise ValueError('LiDAR timestamps must be finite, nonempty and ordered')
    pairings = []

    base_pose_log = np.zeros((seqLength, 6))
    for idx in range(seqLength):

        # Saving Full 3D Data
        (radar_hm, radarBasePose, radarTs) = dataLoader.radarHM(idx)
        if not np.isfinite(radarTs):
            raise ValueError('Radar timestamps must be finite')

        # Frame rates and dropped frames vary. Use the complete timestamp index,
        # not three guessed neighbours around twice the radar index.
        lidarIdx = int(np.argmin(np.abs(lidar_timestamps - radarTs)))
        lidarPts, _, lidarTs = dataLoader.lidarPoints(lidarIdx)
        pairings.append((idx, radarTs, lidarIdx, lidarTs, lidarTs - radarTs))

        np.savez(f"{radarRawPath.resolve().as_posix()}/{idx:03d}", intensity = radar_hm[:, :, :, 0], velocity = radar_hm[:, :, :, 1])
        np.save(f"{lidarRawPath.resolve().as_posix()}/{idx:03d}", lidarPts)

        base_pose_log[idx, :] = se3_logmap(radarBasePose)
        
    np.savetxt(datasetRoot.joinpath("base_pose"), base_pose_log)
    np.savetxt(datasetRoot / "frame_pairing.csv", np.asarray(pairings), delimiter=",",
               header="output_frame,radar_timestamp,lidar_index,lidar_timestamp,delta_seconds", comments="")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a ColoRadar sequence using nearest LiDAR timestamps")
    parser.add_argument('-s', '--seq', type=str, required=True, help='directory of the sequence')
    parser.add_argument('-c', '--calib', type=str, required=True, help='directory of calib data')
    parser.add_argument('-o', '--outputPath', type=str, required=True, help='Output path for dataset generation')
    parser.add_argument('--max-frames', type=int, help='Convert at most N radar frames')
    args = parser.parse_args()
    from utils.ColoRadarDataLoader import CascadeDataLoader
    dataLoader = CascadeDataLoader(args)
    main(dataLoader, args.outputPath, args.max_frames)
