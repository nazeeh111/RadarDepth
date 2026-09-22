import numpy as np
import open3d as o3d
import open3d.core as o3c

if __name__ == "__main__":
    from turbo_colormap import turbo_colormap_data, interpolate
else:
    from .turbo_colormap import turbo_colormap_data, interpolate

# elevation_bins[11]: -16.12 deg
# elevation_bins[20]: 16.12 deg
# Ouster-64 LiDAR fov: -16.8 - 16.5
RADAR_ELEVATION_RANGE = (11, 21)


def load_radar_calib(fname: str, elev_range=RADAR_ELEVATION_RANGE):
    data = np.load(fname)
    range_unit = data['range_bin_width']

    azimuth_lut = data['azimuth_bins'].reshape((1, -1))
    elevation_lut = data['elevation_bins'].reshape(
        (-1, 1))[elev_range[0]:elev_range[1]]

    # Unproject a depth map
    x = np.cos(elevation_lut) @ np.cos(azimuth_lut)
    y = np.cos(elevation_lut) @ np.sin(azimuth_lut)
    z = np.tile(np.sin(elevation_lut), (1, 128))

    r = np.arange(1, 128 + 1).reshape((1, 1, -1)) * range_unit

    xr = np.expand_dims(x, axis=-1) * r
    yr = np.expand_dims(y, axis=-1) * r
    zr = np.expand_dims(z, axis=-1) * r

    xyz = np.stack((xr.flatten(), yr.flatten(), zr.flatten()), axis=-1)
    return xyz


def colorize(intensity: np.ndarray, colormap='turbo'):
    # Note linear normalization might not be appropriate
    min_intensity = intensity.min()
    max_intensity = intensity.max()
    normalized_intensity = (intensity - min_intensity) / max_intensity

    if colormap == 'gray':
        return np.stack(
            (normalized_intensity, normalized_intensity, normalized_intensity),
            axis=-1)
    elif colormap == 'turbo':
        colormap = np.array(turbo_colormap_data)
        return interpolate(colormap, normalized_intensity)


def load_radar_as_pcd(fname: str,
                      xyz: np.ndarray,
                      elev_range=RADAR_ELEVATION_RANGE,
                      use_cfar_image = False):
    data = np.load(fname)
    intensity = data['intensity'].astype(np.float32)
    intensity = intensity[elev_range[0]:elev_range[1]]

    if use_cfar_image:
        cfar = CFAR(guardCell=0, traincell=1, squareFilter = False, faRate=2.5e-2, addTrain = 3, addGuard = 6)
        cfar_intensity = np.zeros_like(intensity)
        for i in range(len(intensity)):
            cfar_intensity[i] =  cfar.normalize_image(cfar.genImage(intensity[i]))
        
        intensity = cfar_intensity

    heat = o3d.core.Tensor(colorize(intensity.flatten()),
                           dtype=o3c.Dtype.Float32)
    tpcd = o3d.t.geometry.PointCloud(
        o3d.core.Tensor(xyz.astype(np.float32), dtype=o3c.Dtype.Float32))
    tpcd.point['colors'] = heat
    return tpcd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('radar_npz')
    parser.add_argument('radar_calib_npz')
    args = parser.parse_args()

    xyz = load_radar_calib(args.radar_calib_npz)
    tpcd = load_radar_as_pcd(args.radar_npz, xyz)
    print(tpcd.point['positions'].shape, tpcd.point['colors'].shape)
    o3d.visualization.draw([tpcd])
