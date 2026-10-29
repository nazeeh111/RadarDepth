"""Recompute synthetic label conversion and geometry; no trained inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_dataset.numpy_to_trainset import convert
from utils.pairing import radarimg2lidar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New, nonexisting output folder')
    parser.add_argument('--distance', type=float, default=2.5, help='Synthetic nearest source range in metres (0, 7)')
    args = parser.parse_args(argv)
    if not np.isfinite(args.distance) or not 0 < args.distance < 7:
        parser.error('--distance must be finite and between 0 and 7 metres')
    if args.output.exists():
        parser.error('--output must not exist; prior results are preserved')
    raw = args.output / 'raw'
    (raw / 'lidar').mkdir(parents=True)
    (raw / 'radar').mkdir()
    mapping = ROOT / 'calib/radarMapping.npz'
    with np.load(mapping) as calib:
        e = float(calib['elevation_bins'][13])
        a = float(calib['azimuth_bins'][60])
        range_width = float(calib['range_bin_width'])
    ray = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    points = np.array([ray*args.distance, ray*(args.distance+1), [-3,0,0]])
    np.save(raw / 'lidar/000.npy', points)
    # An original ideal heatmap fixture, not measured radar or learned depth.
    volume = np.zeros((32,128,128), dtype=np.float32)
    range_index = int(np.clip(round(args.distance/range_width)-1,0,127))
    volume[13,60,range_index] = 1
    np.savez(raw / 'radar/000.npz', intensity=volume, velocity=np.zeros_like(volume))
    rig = args.output / 'identity-rig.npz'
    np.savez(rig, T_bl=np.eye(4), T_br=np.eye(4))
    converted = args.output / 'converted'
    convert(raw/'lidar', raw/'radar', mapping, rig, converted, max_frames=1)
    depth = np.load(converted / 'lidar/00000.npy')
    valid = depth != 10
    masked = np.where(valid,depth,0)
    xyz = radarimg2lidar(mapping,masked)[valid.ravel()]
    expected = ray*args.distance
    error = float(np.max(np.abs(xyz-expected)))
    if xyz.shape != (1,3) or error > 1e-10:
        raise RuntimeError('Geometry disagrees with analytic spherical reference')
    ply = args.output / 'geometry.ply'
    ply.write_text('ply\nformat ascii 1.0\ncomment synthetic geometry, metres, x forward y left z up\nelement vertex 1\nproperty double x\nproperty double y\nproperty double z\nend_header\n' + ' '.join(map(str,xyz[0]))+'\n')
    report = {'evidence':'real conversion/projection algorithm on original ideal synthetic input; not trained inference',
              'distance_m':args.distance,'valid_depth_pixels':int(valid.sum()),
              'geometry_xyz_m':xyz.tolist(),'maximum_coordinate_error_m':error,
              'rejected_rear_points':1, 'nearest_return_wins':True,
              'calibration_sha256':hashlib.sha256(mapping.read_bytes()).hexdigest(),
              'outputs_sha256':{str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(args.output.rglob('*')) if p.is_file()}}
    (args.output / 'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
