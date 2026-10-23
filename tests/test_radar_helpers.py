"""Exact numerical helpers without loading optional GPU, GTSAM or Open3D imports.

These are synthetic helper checks, not validation of trained velocity inference.
"""
import ast
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative_path, names, namespace):
    tree = ast.parse((ROOT / relative_path).read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, relative_path, 'exec'), namespace)
    return namespace


class RadarHelpers(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.calibration = str(Path(self.temporary.name) / 'calib.npz')
        np.savez(self.calibration, range_bin_width=0.5,
                 azimuth_bins=np.linspace(-0.7, 0.7, 128),
                 elevation_bins=np.linspace(-0.4, 0.4, 32))
        projection = load_functions('utils/pairing.py', {'radarimg2lidar'},
                                    {'np': np, 'RADAR_ELEVATION_RANGE': (8, 24)})
        self.velocity = load_functions(
            'utils/velocity.py', {'depth_map_to_velocity_measurement', 'filter_null_pts'},
            {'np': np, 'radarimg2lidar': projection['radarimg2lidar']}
        )['depth_map_to_velocity_measurement']
        turbo = {}
        exec(compile((ROOT / 'utils/turbo_colormap.py').read_text(),
                     'utils/turbo_colormap.py', 'exec'), turbo)
        self.palette = np.array(turbo['turbo_colormap_data'])
        self.colorize = load_functions('utils/radar.py', {'colorize'},
            {'np': np, 'interpolate': turbo['interpolate'],
             'turbo_colormap_data': turbo['turbo_colormap_data']})['colorize']

    def volume(self):
        r, e, a = np.indices((128, 16, 128))
        return 10000*r + 100*e + a

    def test_range_and_azimuth_are_independent(self):
        depth = np.zeros((16, 128)); depth[2, 3] = 3.5
        confidence = np.zeros_like(depth); confidence[2, 3] = 0.6
        result = self.velocity(depth, self.volume(), confidence, self.calibration)
        self.assertEqual(result.shape, (1, 5))
        self.assertEqual(result[0, 3], 70203)
        self.assertEqual(result[0, 4], 0.6)
        self.assertAlmostEqual(np.linalg.norm(result[0, :3]), 3.5)

    def test_confidence_uses_depth_pixels_even_when_zero(self):
        depth = np.zeros((16, 128)); depth[1, 5] = 2.; depth[3, 9] = 4.
        confidence = np.zeros_like(depth)
        confidence[3, 9] = .8; confidence[0, 0] = .9
        result = self.velocity(depth, self.volume(), confidence, self.calibration)
        np.testing.assert_allclose(result[:, 3:], [[40105, 0.], [80309, .8]])
        np.testing.assert_allclose(np.linalg.norm(result[:, :3], axis=1), [2., 4.])

    def test_clipped_range_and_near_zero_point_share_mask(self):
        depth = np.zeros((16, 128)); depth[1, 1] = -1.; depth[2, 2] = 100.
        depth[3, 3] = 1e-12
        confidence = np.zeros_like(depth); confidence[1, 1] = .2; confidence[2, 2] = .7
        result = self.velocity(depth, self.volume(), confidence, self.calibration)
        self.assertEqual(result.shape, (2, 5))
        np.testing.assert_allclose(result[:, 3:], [[101, .2], [1270202, .7]])

    def test_empty_depth_ignores_unrelated_confidence(self):
        depth = np.zeros((16, 128)); confidence = np.ones_like(depth)
        self.assertEqual(self.velocity(depth, self.volume(), confidence,
                                       self.calibration).shape, (0, 5))

    def test_gray_maps_nonzero_and_negative_ranges(self):
        for values in ([10., 15., 20.], [-20., -15., -10.]):
            colors = self.colorize(np.array(values), 'gray')
            np.testing.assert_allclose(colors, np.repeat([[0.], [.5], [1.]], 3, axis=1))

    def test_constants_are_finite_at_lower_palette_endpoint(self):
        with np.errstate(divide='raise', invalid='raise'):
            for value in (0., 8., -4.):
                values = np.full((2, 3), value)
                np.testing.assert_array_equal(self.colorize(values, 'gray'), np.zeros((2, 3, 3)))
                np.testing.assert_allclose(self.colorize(values, 'turbo'),
                                           np.broadcast_to(self.palette[0], (2, 3, 3)))

    def test_turbo_endpoints_and_zero_based_compatibility(self):
        np.testing.assert_allclose(self.colorize(np.array([10., 20.]), 'turbo'), self.palette[[0, -1]])
        np.testing.assert_allclose(self.colorize(np.array([0., .5, 1.]), 'gray'),
                                   np.repeat([[0.], [.5], [1.]], 3, axis=1))


if __name__ == '__main__':
    unittest.main()
