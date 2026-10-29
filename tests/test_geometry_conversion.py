"""Production NumPy projection/conversion, compared with scalar geometry."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.pairing import lidar2radar, radarimg2lidar
from generate_dataset.numpy_to_trainset import convert


class GeometryConversion(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapping = ROOT / 'calib/radarMapping.npz'
        with np.load(self.mapping) as data:
            self.az = data['azimuth_bins'].ravel()
            self.el = data['elevation_bins'].ravel()[8:24]
            self.full_el = data['elevation_bins'].ravel()
        self.rig = self.root / 'rig.npz'
        np.savez(self.rig, T_bl=np.eye(4), T_br=np.eye(4))

    def ray(self, row, col, radius):
        e, a = float(self.el[row]), float(self.az[col])
        return np.array([radius*np.cos(e)*np.cos(a), radius*np.cos(e)*np.sin(a), radius*np.sin(e)])

    def reference(self, points, transform):
        # Independent scalar implementation, including nearest visible return.
        import math
        out = np.full((16, len(self.az)), 10.)
        occupied = set()
        for point in points:
            if not np.isfinite(point[:3]).all(): continue
            x, y, z = transform[:3, :3].dot(point[:3]) + transform[:3, 3]
            radius = math.sqrt(x*x+y*y+z*z)
            if radius == 0: continue
            a, e = math.atan2(y,x), math.atan2(z,math.hypot(x,y))
            if not self.az.min() <= a <= self.az.max() or not self.el.min() <= e <= self.el.max():continue
            col = min(range(len(self.az)), key=lambda i:abs(float(self.az[i])-a))
            row = min(range(len(self.el)), key=lambda i:abs(float(self.el[i])-e))
            out[row,col] = min(out[row,col],radius) if (row,col) in occupied else radius
            occupied.add((row,col))
        return out

    def test_nearest_return_fov_nonfinite_and_order(self):
        points=np.array([self.ray(5,60,2),self.ray(5,60,5),[-3,0,0],[0,0,0],[np.nan,0,0],self.ray(8,80,12)])
        expected=self.reference(points,np.eye(4))
        np.testing.assert_allclose(lidar2radar(self.mapping,np.eye(4),points),expected,atol=1e-12)
        np.testing.assert_allclose(lidar2radar(self.mapping,np.eye(4),points[::-1]),expected,atol=1e-12)
        self.assertAlmostEqual(expected[5,60],2)
        self.assertAlmostEqual(expected[8,80],12)
        self.assertEqual(np.count_nonzero(expected!=10),2)

    def test_rigid_transform_and_unprojection_units_axes(self):
        T=np.eye(4);T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];T[:3,3]=[.1,-.3,.2]
        radar=np.array([self.ray(6,44,3),self.ray(10,90,4)])
        lidar=(radar-T[:3,3])@T[:3,:3]
        depth=lidar2radar(self.mapping,T,lidar)
        expected=self.reference(lidar,T)
        np.testing.assert_allclose(depth,expected,atol=1e-12)
        depth[depth==10]=0
        xyz=radarimg2lidar(self.mapping,depth)
        np.testing.assert_allclose(xyz[[6*128+44,10*128+90]],radar,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(xyz,axis=1).reshape(16,128),depth,atol=1e-12)

    def fixture(self):
        lidar=self.root/'raw/lidar';radar=self.root/'raw/radar'
        lidar.mkdir(parents=True);radar.mkdir()
        volume=np.arange(128*32*128,dtype=np.float32).reshape(32,128,128)
        for stem,radius in [('000',2.),('001',3.)]:
            np.save(lidar/f'{stem}.npy',np.array([self.ray(5,60,radius),self.ray(5,60,radius+2),[-3,0,0]]))
            np.savez(radar/f'{stem}.npz',intensity=volume,velocity=-volume)
        return lidar,radar,volume

    def test_actual_cli_from_another_cwd_bounded_and_changed_input(self):
        lidar,radar,volume=self.fixture()
        command=[sys.executable,str(ROOT/'generate_dataset/numpy_to_trainset.py'),'-l',str(lidar),'-r',str(radar),'-m',str(self.mapping),'-c',str(self.rig),'--max-frames','1']
        out=self.root/'out'
        result=subprocess.run(command+['-o',str(out)],cwd=self.root,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout)['converted_frames'],1)
        with np.load(out/'radar/00000.npz') as data:
            np.testing.assert_array_equal(data['intensity'],volume[8:24].transpose(2,0,1))
            np.testing.assert_array_equal(data['velocity'],-volume[8:24].transpose(2,0,1))
        self.assertAlmostEqual(np.load(out/'lidar/00000.npy')[5,60],2)
        np.save(lidar/'000.npy',np.array([self.ray(5,60,3.25)]))
        out2=self.root/'changed'
        subprocess.run(command+['-o',str(out2)],cwd=self.root,capture_output=True,text=True,check=True)
        self.assertAlmostEqual(np.load(out2/'lidar/00000.npy')[5,60],3.25)
        failed=subprocess.run(command+['-o',str(out)],capture_output=True,text=True)
        self.assertEqual(failed.returncode,2)
        self.assertIn('will not be overwritten',failed.stderr)

    def test_pairing_mismatch_fails_before_writing(self):
        lidar,radar,_=self.fixture()
        (lidar/'001.npy').rename(lidar/'002.npy')
        with self.assertRaisesRegex(ValueError,'matching nonempty frame stems'):
            convert(lidar,radar,self.mapping,self.rig,self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_raw_conversion_uses_timestamp_not_assumed_frame_ratio(self):
        from generate_dataset.raw_dataset_to_numpy import main
        test = self
        class SyntheticLoader:
            plot_data = [0, 1]
            lidar_plot_data = [(0.2,), (0.7,), (0.9,)]
            lidar_params = {'T_bs': np.eye(4)}
            radar_params = {'T_bs': np.eye(4), 'range_bin_width': .0592944,
                            'azimuth_bins': test.az, 'elevation_bins': test.full_el}
            def radarHM(self, index):
                return np.zeros((32,128,128,2),dtype=np.float32), np.eye(4), [.3,.81][index]
            def lidarPoints(self, index):
                return np.array([[index+1.,0,0,1]]), np.eye(4), self.lidar_plot_data[index][0]
        out=self.root/'raw-converted'
        main(SyntheticLoader(),str(out))
        np.testing.assert_array_equal(np.load(out/'raw/lidar/000.npy'),[[1.,0,0,1]])
        np.testing.assert_array_equal(np.load(out/'raw/lidar/001.npy'),[[3.,0,0,1]])
        pairings=np.loadtxt(out/'frame_pairing.csv',delimiter=',',skiprows=1)
        np.testing.assert_array_equal(pairings[:,2],[0,2])
        np.testing.assert_allclose(pairings[:,4],[-.1,.09])

    def test_invalid_shape_slice_and_empty_points(self):
        np.testing.assert_array_equal(lidar2radar(self.mapping,np.eye(4),np.empty((0,3))),np.full((16,128),10.))
        for points,T,slice_ in [(np.zeros((3,2)),np.eye(4),(8,24)),(np.zeros((3,3)),np.eye(3),(8,24)),(np.zeros((3,3)),np.eye(4),(24,8))]:
            with self.assertRaises(ValueError):lidar2radar(self.mapping,T,points,elev_range=slice_)


if __name__=='__main__':unittest.main()
