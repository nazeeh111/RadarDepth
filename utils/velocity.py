from turtle import distance
from typing import Dict
import gtsam
import numpy as np
from .pairing import radarimg2lidar
import torch

def filter_null_pts(pts : np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(pts, axis=1)
    mask = np.bitwise_not(np.isclose(distance, 0))
    return pts[mask, :]


def inferVelocity(pose1 : gtsam.Pose3, pose2 : gtsam.Pose3, pose3: gtsam.Pose3, dt1, dt2) -> np.ndarray:
    bt = pose1.between(pose3)
    twist = gtsam.Pose3.Logmap(bt)
    meanV = twist[3:]
    dt = dt2 - dt1
    return meanV / dt

def depth_map_to_velocity_measurement(depth_map : np.ndarray,\
        velocity_volume : np.ndarray, confidence_volume : np.ndarray,\
        calib_fname : str) -> np.ndarray:
    '''
    With Radar Depthmap, find out all the velocity measurements in the velocity volume

    returns a n x 5 velocity measurement in the sensor frame
    (x, y, z, doppler_value, confidence)
    '''

    (h, _) = depth_map.shape
    lower_range = int(16 - h / 2)
    calib_data = np.load(calib_fname)
    range_unit = calib_data['range_bin_width']

    e, a = np.nonzero(depth_map)
    r = np.round(depth_map[e, a] / range_unit).astype(int)
    r = np.clip(a, a_min=0, a_max=127)

    # confidence = confidence_volume.flatten().reshape((-1, 1))
    h, w = np.nonzero(confidence_volume)
    confidence = confidence_volume[h, w].reshape((-1, 1))
    # e = e + lower_range

    velocity_values = velocity_volume[r, e, a].reshape((-1, 1))

    points = radarimg2lidar(calib_fname, depth_map)
    points = filter_null_pts(points)
    points = np.hstack((points, velocity_values, confidence))

    return points

def point_velocity_vector_filtering(pts : np.ndarray, body_vel : np.ndarray) -> np.ndarray:
    '''
    pts : (N, 4), N Detections, x, y, z, coordinates in body frame, v measured velocity value
    body_vel : (3, ), velocity in the body frame
    '''
    
    # Should discard z-axis values
    
    bv = body_vel.copy()
    bv[2] = 0
    
    vel_vector = np.zeros((len(pts), 3))
    vel_vector = pts[:, :3] / np.linalg.norm(pts[:, :3], axis=1)[:, np.newaxis]
    
    err = bv @ vel_vector.T + pts[:, 3]
    return err

def clip_cov_mat(cov : np.ndarray) -> np.ndarray:
    diag_entry = np.diagonal(cov)
    cliped_value = np.clip(diag_entry, a_min=0.005, a_max=None)
    return np.diag(cliped_value)

def getVelocityMeasurement(ptsWithVel : np.ndarray, initial = np.zeros(3), weight=False) -> np.ndarray:
    '''
    ptsWithVel = n x 4, index 0,1,2 is position [x,y,z], index 3 is doppler velocity
    '''

    (n, d) = ptsWithVel.shape
    if d < 5 and weight:
        raise Exception("Weighted but no confidence")

    if weight:
        # weight = 1 / (1 + np.exp(-10 * (ptsWithVel[:, 4].copy() - 0.5)))
        weight = ptsWithVel[:, 4].copy()
        ptsWithVel = ptsWithVel[:, :4]
    else:
        weight = np.ones(len(ptsWithVel))
    
    if not np.allclose(initial, np.zeros(3)):
        # Given a baseline
        err = point_velocity_vector_filtering(ptsWithVel, initial)
        mask = np.abs(err) < 0.5
        ptsWithVel = ptsWithVel[mask, :]
        weight = weight[mask]

    # if len(ptsWithVel) < 5:
    #     return initial, np.eye(3) * 10

    bVec = ptsWithVel[:, 3]
    unitVec = ptsWithVel[:, :3] / np.linalg.norm(ptsWithVel[:, :3], axis=1)[:, np.newaxis]
    # unitVec = unitVec * weight[:, np.newaxis]

    A = unitVec.T @ np.diag(weight) @ unitVec
    b = unitVec.T @ np.diag(weight) @ bVec

    if np.isclose(np.linalg.det(A), 0):
        A += 1e-2 * np.eye(3)

    x = np.linalg.solve(A, b)

    cov = (0.254 / np.sqrt(12)) ** 2

    try:
        ainv = np.linalg.pinv(A / (cov * len(unitVec)))
    except:
        print(f"pinv failed, assigning identity")
    #     print(f"A mat: {A}")
    #     print(f"unitVec: {unitVec}")
    #     print(f"Solved x: {x}")
        ainv = np.eye(3) * 0.01
    
    return x, clip_cov_mat(ainv)

def updates(unitVec, b, initial_val):
    '''
    '''
    res = initial_val
    lastres = np.zeros_like(initial_val)
    max_iter = 15
    iter_count = 0
    while not np.allclose(res, lastres) and iter_count < max_iter:
        # res = lastres.copy()
        lastres = res.copy()
        res = mestimator(unitVec, b, lastres)
        iter_count += 1
        # print(f"res: {res}, lastres: {lastres}")
    return res

def mestimator(unitVec, b, initial_val):
    residual = (unitVec @ initial_val - b) ** 2
    # print(np.sum(residual))
    residual *= 100
    mask = residual > 1
    residual[mask] = 1 / residual[mask]
    # residual = 1 / residual

    Amat = unitVec * residual[:, np.newaxis]
    bmat = b * residual
    # print(np.sum(residual))
    return np.linalg.solve(Amat.T @ Amat, Amat.T @ bmat)

def infer_velocity_from_network_output(output_dict : Dict, velocity_volume : np.ndarray, calib_npz : str, initial=np.zeros(3)) -> np.ndarray:
    '''
    '''
    output, volume = output_dict['depth'], output_dict['prob_volume']
    output = output.squeeze(0).cpu().detach().numpy().copy()
    volume = volume.squeeze(0).cpu().detach().numpy().copy()
    mask = output_dict['segment_mask']
    if mask is not None:
        mask = mask.squeeze(0).cpu().detach().sigmoid().numpy().copy()
        output[mask[0, :, :] < 0.5] = 0

    v_max = np.max(volume)
    v_min = np.min(volume)
    volume = (volume - v_min) / (v_max - v_min) + 0.01
    # Remove poor resolution on the side
    # volume[:, 0:16] = 0
    # volume[:, 112:] = 0
    
    # Remove low confidence points
    # output[volume < 0.1] = 0
    # volume[volume < 0.1] = 0

    # Remove too close points
    distance_mask = output < 1
    output[distance_mask] = 0
    volume[distance_mask] = 0

    velocity_vector = depth_map_to_velocity_measurement(output, velocity_volume, volume, calib_npz)

    # import pyvista as pv
    # p = pv.Plotter()
    # p.add_mesh(pv.PolyData(velocity_vector[:, :3]))
    # p.show()
    
    velocity_vector[:, 3] = velocity_vector[:, 3] - 0.254

    # velocity_vector[:, 3] = -1
    # velocity_vector[:, 4] = 1
    # print(velocity_vector)

    thisv, cov = getVelocityMeasurement(velocity_vector, initial=initial, weight=False)
    thisv = -thisv

    return thisv, cov


def infer_ground_truth_velocity(base_pose : np.ndarray, T_br : gtsam, index : int, dt):
    this_radar_pose = gtsam.Pose3.Expmap(base_pose[index, :]).compose(T_br)
    lastBasePose = gtsam.Pose3.Expmap(base_pose[index - 1, :]).compose(T_br)
    nextRadarPose = gtsam.Pose3.Expmap(base_pose[index + 1, :]).compose(T_br)

    if index == 0:
        v = inferVelocity(this_radar_pose, this_radar_pose, nextRadarPose, 0, 0 + dt / 2)
    else:
        v = inferVelocity(lastBasePose, this_radar_pose, nextRadarPose, 0, 0 + dt)
    return v

