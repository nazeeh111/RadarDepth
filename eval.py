"""Headless checkpoint evaluation, with an explicitly requested optional viewer.

This executes a supplied checkpoint; neither its existence nor a successful run
establishes that the weights were trained or that input data is measured.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import time
from typing import List

import numpy as np

from utils.pairing import radarimg2lidar


def load_raw_lidar(path_to_lidar: str, T_rl: np.ndarray, frame_stems=None) -> List[np.ndarray]:
    """Load authorized xyz files in explicit dataset order when supplied."""
    transform = np.asarray(T_rl)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('T_rl must be a finite 4x4 transform')
    paths = sorted(Path(path_to_lidar).glob('*.npy'))
    if frame_stems is not None:
        key = lambda stem: int(stem) if stem.isdecimal() else stem
        mapping = {}
        for path in paths:
            value = key(path.stem)
            if value in mapping:
                raise ValueError('Raw LiDAR has duplicate numeric frame stems')
            mapping[value] = path
        try:
            paths = [mapping[key(stem)] for stem in frame_stems]
        except KeyError as error:
            raise ValueError(f'Missing raw LiDAR for dataset frame {error.args[0]}') from error
    ret_val = []
    for path in paths:
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
            raise ValueError(f'Raw LiDAR must be finite Nx3+ coordinates: {path}')
        points = np.asarray(points[:, :3], dtype=float)
        ret_val.append(points @ transform[:3, :3].T + transform[:3, 3])
    return ret_val


def infer_frame(model, radar, threshold=0.4):
    """Run the real CPU model in evaluation/inference mode; preserve caller mode."""
    import torch
    radar = np.asarray(radar)
    if (radar.dtype.kind not in 'buif' or radar.ndim != 3 or radar.shape[0] != 128 or any(n % 8 for n in radar.shape[1:])
            or min(radar.shape[1:]) < 8 or not np.isfinite(radar).all()):
        raise ValueError('Radar must be finite [128, elevation, azimuth], with spatial sizes divisible by 8')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Mask threshold must be in [0,1]')
    parameter = next(model.parameters())
    if parameter.device.type != 'cpu':
        raise ValueError('Headless evaluation currently supports CPU models')
    sample = torch.as_tensor(radar, dtype=parameter.dtype).unsqueeze(0).unsqueeze(0)
    was_training = model.training
    started = time.perf_counter()
    try:
        model.eval()
        with torch.inference_mode():
            result = model(sample)
            depth = result['depth'].squeeze(0).cpu().numpy().copy()
            logits = result['segment_mask']
            if logits is None:
                probability = np.ones_like(depth)
            else:
                if tuple(logits.shape) != (1, 2, *depth.shape):
                    raise ValueError('Expected two segmentation logits [1,2,elevation,azimuth]')
                # Training uses two binary target channels, valid then invalid.
                probability = torch.softmax(logits, dim=1)[0, 0].cpu().numpy().copy()
    finally:
        model.train(was_training)
    if depth.shape != radar.shape[1:] or not np.isfinite(depth).all() or not np.isfinite(probability).all():
        raise ValueError('Model produced invalid depth/probability arrays')
    selected = (probability >= threshold) & (depth > 0)
    return depth, probability, selected, time.perf_counter() - started


def depth_metrics(depth, label, selected):
    """Metrics on all valid depth labels, with mask coverage reported separately."""
    from networks.DepthRefinement.network_parts import R_WIDTH
    label = np.asarray(label)
    if label.dtype.kind not in 'buif' or label.shape != depth.shape or not np.isfinite(label).all():
        raise ValueError('Label must be finite and match the predicted angular grid')
    valid = (label > 0) & (label < R_WIDTH * 128)
    errors = depth[valid] - label[valid]
    return {'valid_depth_pixels': int(valid.sum()),
            'depth_mae_m': float(np.mean(np.abs(errors))) if errors.size else None,
            'depth_rmse_m': float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            'selected_pixels': int(selected.sum()),
            'selected_valid_pixels': int((selected & valid).sum()),
            'true_positive': int((selected & valid).sum()),
            'false_positive': int((selected & ~valid).sum()),
            'false_negative': int((~selected & valid).sum()),
            'true_negative': int((~selected & ~valid).sum())}


def write_ply(path, points):
    with Path(path).open('x') as stream:
        stream.write('ply\nformat ascii 1.0\ncomment radar frame metres: x forward y left z up\n')
        stream.write(f'element vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n')
        np.savetxt(stream, points, fmt='%.17g')


def process_dataset(model, dataset, radar_calib_npz, corresponding_lidar_pts=None,
                    output_dir=None, threshold=0.4, max_frames=None, viewer=False):
    """Evaluate actual frames, returning metrics and optionally writing new outputs."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Mask threshold must be in [0,1]')
    if max_frames is not None and max_frames < 1:
        raise ValueError('max_frames must be positive')
    count = len(dataset) if max_frames is None else min(len(dataset), max_frames)
    if count < 1:
        raise ValueError('Evaluation dataset is empty')
    if corresponding_lidar_pts is not None and len(corresponding_lidar_pts) != count:
        raise ValueError('Raw LiDAR count must match the selected dataset frames')
    from networks.DepthRefinement.network_parts import R_WIDTH
    with np.load(radar_calib_npz, allow_pickle=False) as calibration:
        elevation = np.asarray(calibration['elevation_bins'], dtype=float).ravel()[8:24]
        azimuth = np.asarray(calibration['azimuth_bins'], dtype=float).ravel()
        width = np.asarray(calibration['range_bin_width'], dtype=float)
    expected_shape = (len(elevation), len(azimuth))
    if expected_shape != (16, 128):
        raise ValueError('Current checkpoint/geometry contract requires 16 selected elevation and 128 azimuth bins')
    if (width.size != 1 or not np.isfinite(width).all()
            or not np.isclose(float(width.ravel()[0]), R_WIDTH, rtol=1e-6, atol=1e-9)):
        raise ValueError('Calibration range_bin_width must match the fixed DRNet3D 0.0592944 metre bins')
    for name, angles, limit in [('azimuth', azimuth, np.pi), ('elevation', elevation, np.pi/2)]:
        if (not np.isfinite(angles).all() or (np.abs(angles) > limit).any()
                or not (np.all(np.diff(angles) > 0) or np.all(np.diff(angles) < 0))):
            raise ValueError(f'Calibration {name} bins must be finite strictly monotonic radians')
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    if viewer:
        import rerun as rr
        rr.init('PCD Comparison')
        rr.spawn()
    records = []
    for index in range(count):
        data = dataset[index]
        input_paths = (Path(dataset.radar_fnames[index]), Path(dataset.lidar_fnames[index]))
        input_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
        depth, probability, selected, elapsed = infer_frame(model, data['radar'], threshold)
        if depth.shape != expected_shape:
            raise ValueError('Dataset angular shape does not match calibration')
        metrics = depth_metrics(depth, data['lidar'], selected)
        masked_depth = np.where(selected, depth, 0)
        xyz = radarimg2lidar(radar_calib_npz, masked_depth)[selected.ravel()]
        if not np.isfinite(xyz).all():
            raise ValueError('Projection produced nonfinite coordinates')
        stem = Path(dataset.radar_fnames[index]).stem
        record = dict(frame=stem, inference_seconds=elapsed, points=int(len(xyz)), metrics=metrics,
                      inputs_sha256=input_hashes)
        if destination is not None:
            array_path = destination / f'{stem}.npz'
            np.savez(array_path, depth_m=depth, masked_depth_m=masked_depth,
                     validity_probability=probability, selected_mask=selected,
                     label_depth_m=data['lidar'])
            point_path = destination / f'{stem}.ply'
            write_ply(point_path, xyz)
            record['sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (array_path, point_path)}
        records.append(record)
        if viewer:
            rr.set_time('frame', sequence=index)
            rr.log('world/radar', rr.Points3D(positions=xyz, colors=[255, 0, 0], radii=0.02))
            label = np.asarray(data['lidar'])
            valid = (label > 0) & (label < 0.0592944 * 128)
            label_xyz = radarimg2lidar(radar_calib_npz, np.where(valid, label, 0))[valid.ravel()]
            rr.log('world/label_lidar', rr.Points3D(positions=label_xyz, colors=[0, 255, 0], radii=0.02))
            if corresponding_lidar_pts is not None:
                rr.log('world/raw_lidar', rr.Points3D(positions=corresponding_lidar_pts[index], colors=[0, 0, 255], radii=0.02))
    report = {'evidence': 'supplied-checkpoint numerical evaluation; training and measured-data provenance are external',
              'axes': ['range', 'elevation', 'azimuth'], 'depth_units': 'metres',
              'coordinate_frame': 'radar: x forward, y left, z up',
              'mask_threshold': threshold, 'mask_semantics': 'two-class softmax, class0 valid',
              'depth_metrics': 'all valid label pixels, independent of predicted segmentation selection',
              'frames': records, 'numpy_version': np.__version__}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_path', required=True, help='Compatible DRNet3D state_dict checkpoint')
    parser.add_argument('-c', '--radar_calib_npz', required=True)
    parser.add_argument('-d', '--dataset_dir', required=True, help='Paired radar/ and lidar/ converted arrays')
    parser.add_argument('-r', '--rig_calibration_npz', help='Optional extrinsic rig calibration for raw LiDAR viewer')
    parser.add_argument('-l', '--lidar_dir', help='Optional raw LiDAR directory, matched by numeric frame stem')
    parser.add_argument('-o', '--output', required=True, help='New result directory; never overwrites prior results')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--threads', type=int, default=1, help='Positive CPU thread bound, default 1')
    parser.add_argument('--viewer', action='store_true', help='Explicitly launch optional Rerun viewer')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if bool(args.rig_calibration_npz) != bool(args.lidar_dir):
        parser.error('--rig_calibration_npz and --lidar_dir must be supplied together')
    try:
        import torch
        from dataloader.dataset import Dataset
        from networks.DepthRefinement.model import DRNet3D, DEPTH_VAL
        torch.set_num_threads(args.threads)
        checkpoint = Path(args.model_path)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError('Checkpoint must be the state_dict written by train.py')
        model = DRNet3D(DEPTH_VAL, segment=any(k.startswith('segmentation.') for k in state))
        model.load_state_dict(state, strict=True)
        model.double().eval()
        dataset = Dataset(args.dataset_dir)
        raw = None
        if args.lidar_dir:
            with np.load(args.rig_calibration_npz, allow_pickle=False) as rig:
                T_rl = np.linalg.inv(rig['T_br']) @ rig['T_bl']
            selected = dataset.radar_fnames[:args.max_frames]
            raw = load_raw_lidar(args.lidar_dir, T_rl, [Path(p).stem for p in selected])
        result = process_dataset(model, dataset, args.radar_calib_npz, raw, args.output,
                                 args.threshold, args.max_frames, args.viewer)
        result['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result['checkpoint_path'] = str(checkpoint.resolve())
        result['dataset_path'] = str(Path(args.dataset_dir).resolve())
        result['calibration_sha256'] = hashlib.sha256(Path(args.radar_calib_npz).read_bytes()).hexdigest()
        result['torch_version'] = torch.__version__
        result['cpu_threads'] = args.threads
        (Path(args.output) / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, pickle.UnpicklingError, EOFError) as error:
        parser.exit(2, f'Evaluation failed: {error}\n')
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
