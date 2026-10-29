"""Convert matched raw radar/LiDAR NumPy frames using only NumPy."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pairing import lidar2radar, RADAR_ELEVATION_RANGE


def convert(lidar_path, radar_path, radar_calib, rig_calib, output,
            elev_range=RADAR_ELEVATION_RANGE, max_frames=None, image_gen=False):
    lidar_path, radar_path, output = map(Path, (lidar_path, radar_path, output))
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if not lidar_path.is_dir() or not radar_path.is_dir():
        raise ValueError("LiDAR and radar input directories must exist")
    radar_files = {p.stem: p for p in radar_path.glob('*.npz')}
    lidar_files = {p.stem: p for p in lidar_path.glob('*.npy')}
    if not radar_files or radar_files.keys() != lidar_files.keys():
        raise ValueError("Radar and LiDAR must have matching nonempty frame stems")
    stems = sorted(radar_files)
    if any(not stem.isdecimal() for stem in stems):
        raise ValueError("Frame stems must be decimal integers")
    stems.sort(key=int)
    output_stems = [f'{int(stem):05d}' for stem in stems]
    if len(set(output_stems)) != len(output_stems):
        raise ValueError("Frame stems map to duplicate numeric output names")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing data will not be overwritten")
    with np.load(rig_calib, allow_pickle=False) as calib:
        T_rl = np.linalg.inv(calib['T_br']) @ calib['T_bl']
    with np.load(radar_calib, allow_pickle=False) as calib:
        expected_angles = (len(calib['elevation_bins']), len(calib['azimuth_bins']))
    lower, upper = elev_range
    if not 0 <= lower < upper <= expected_angles[0]:
        raise ValueError("Elevation slice is outside calibration")
    output.mkdir(parents=True, exist_ok=True)
    (output / 'radar').mkdir(exist_ok=True)
    (output / 'lidar').mkdir(exist_ok=True)
    records = []
    for stem in stems[:max_frames]:
        with np.load(radar_files[stem], allow_pickle=False) as radar:
            intensity, velocity = radar['intensity'], radar['velocity']
            if (intensity.ndim != 3 or intensity.shape != velocity.shape
                    or intensity.shape[:2] != expected_angles
                    or not np.isfinite(intensity).all() or not np.isfinite(velocity).all()):
                raise ValueError(f"Invalid radar frame {stem}: expected finite (elevation, azimuth, range) arrays")
            intensity = intensity[lower:upper].transpose(2, 0, 1)
            velocity = velocity[lower:upper].transpose(2, 0, 1)
        lidar = np.load(lidar_files[stem], allow_pickle=False)
        depth = lidar2radar(radar_calib, T_rl, lidar, elev_range=elev_range)
        name = f'{int(stem):05d}'
        np.savez(output / 'radar' / f'{name}.npz', intensity=intensity, velocity=velocity)
        np.save(output / 'lidar' / f'{name}.npy', depth)
        if image_gen:
            from PIL import Image
            (output / 'img').mkdir(exist_ok=True)
            span = np.ptp(depth)
            normalized = (depth - depth.min()) / span if span else np.zeros_like(depth)
            Image.fromarray((normalized * 255).astype(np.uint8)).save(output / 'img' / f'{name}.jpg')
        records.append({'input_stem': stem, 'output_stem': name})
    manifest = {'frames': records, 'elevation_range': list(elev_range),
                'radar_axes': ['range', 'elevation', 'azimuth'], 'depth_units': 'metres',
                'missing_depth_m': 10.0, 'T_rl': T_rl.tolist(),
                'radar_calibration': str(Path(radar_calib).resolve()),
                'rig_calibration': str(Path(rig_calib).resolve())}
    (output / 'conversion.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def boolean_flag(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1'):
        return True
    if value.lower() in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--lidar_sequence', required=True)
    parser.add_argument('-r', '--radar_sequence', required=True)
    parser.add_argument('-m', '--radar_calib_npz', required=True)
    parser.add_argument('-c', '--rig_calib_npz', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-i', '--image_gen', nargs='?', const=True, default=False,
                        type=boolean_flag, help='Save JPEGs (requires Pillow); optional true/false')
    parser.add_argument('-lower', '--lower_range', type=int, default=8)
    parser.add_argument('-upper', '--upper_range', type=int, default=24)
    parser.add_argument('--max-frames', type=int, help='Bound the conversion to the first N matched frames')
    args = parser.parse_args(argv)
    try:
        result = convert(args.lidar_sequence, args.radar_sequence, args.radar_calib_npz,
                         args.rig_calib_npz, args.output, (args.lower_range, args.upper_range),
                         args.max_frames, args.image_gen)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Conversion failed: {error}\n')
    print(json.dumps({'converted_frames': len(result['frames']), 'output': args.output}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
