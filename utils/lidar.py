import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def load_lidar_as_xyz_im(fname: str, h=64):
    data = np.load(fname).astype(np.float32)
    # Open3D complains about tensor point cloud not Float32,
    # load as np.float32 fixes this, COULD have other work arounds

    x = data[:, 0].reshape((-1, h))
    y = data[:, 1].reshape((-1, h))
    z = data[:, 2].reshape((-1, h))

    # Need transpose -- stored column-wise
    xyz = np.stack((x, y, z), axis=-1).transpose((1, 0, 2))

    return xyz


def load_lidar_as_pcd(fname: str, h=64):
    xyz = load_lidar_as_xyz_im(fname, h)

    tpcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(xyz.reshape((-1, 3))))
    return tpcd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lidar_npy')
    args = parser.parse_args()

    im = load_lidar_as_xyz_im(args.lidar_npy)
    plt.imshow(im)
    plt.show()

    pcd = load_lidar_as_pcd(args.lidar_npy)
    o3d.visualization.draw([pcd])
