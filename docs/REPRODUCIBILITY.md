# Recomputable subset and research prerequisites

The `geometry-demo` command runs the actual NumPy conversion and projection core.
It creates original ideal inputs, converts the radar axes, resolves LiDAR
occlusions, excludes points outside the selected field of view, unprojects a
new depth label, and exports a PLY plus hashed arrays and JSON. `--distance 3.25`
changes both the source fixture and recomputed output. It does not load DRNet3D,
a trained checkpoint, a camera, or an external recorded result.

## Corrected execution defects

- The previous LiDAR label generator clamped rear/out-of-FOV points to edge pixels
  and let later, farther returns overwrite nearer returns. The current core
  selects the nearest visible return, independent of input order.
- NumPy conversion no longer imports Open3D, SciPy, plotting, or Torch through an
  eager package initializer. It runs from another working directory, validates
  matching frame stems and array shapes, defaults to the network's 16 elevation
  rows (`8:24`), and accepts `--max-frames`. Existing nonempty outputs are refused.
- Raw conversion chooses nearest available LiDAR timestamps rather than assuming
  two LiDAR frames per radar frame. `frame_pairing.csv` records both timestamps
  and signed time offset. No motion compensation, time-offset estimation or
  acceptable synchronization threshold is inferred from nearest-neighbour pairing.
- The training function uses its supplied optimizer. Training and evaluation
  explicitly set network mode; heldout batches no longer update BatchNorm state.
  Loss means are sample-weighted, scheduling uses mean heldout loss, evaluation
  keeps the last incomplete batch, and `--no-plot` permits headless execution.
  The actual DRNet3D CPU test performs two gradient updates and four heldout
  forwards with segmentation enabled; it verifies parameter changes, saved
  weights and exactly two BatchNorm training updates. These are software tests
  with synthetic tensors, not a useful trained radar checkpoint.

## Numerical contracts

Raw heatmaps contain `intensity` and `velocity` arrays with axes
`(elevation, azimuth, range)`; converted arrays use `(range, elevation, azimuth)`.
Calibration `azimuth_bins` and `elevation_bins` are radians.
LiDAR xyz, depth and rig translations are metres. Positive x is forward, positive
y is left, positive z is up. `T_bl` and `T_br` map sensor points to body coordinates;
`T_rl = inverse(T_br) @ T_bl` maps LiDAR to radar. `radarimg2lidar` returns radar-frame
points despite its historical name.

Projection uses the angular bin-centre field of view, including both boundaries;
points beyond it are omitted. Multiple returns use minimum positive radius.
Nonfinite points and zero-radius transformed points are omitted. Empty labels
retain the legacy **10 metre sentinel**, which is outside the model's represented
range. With the bundled calibration the model represents 128 range bins at
0.0592944 metres per bin, approximately 7.59 metres. There is no invented depth
value or learned prediction for unobserved pixels. The existing midpoint
unprojection convention is retained; the bounded example uses the native 16x128
angular grid.

## Full trained path remains unresolved

The official [ColoRadar site](https://arpg.colorado.edu/coloradar/) states Apache
2.0 licensing and distributes runs plus a separate calibration archive. No bulk
run was downloaded for this repair. Use authorized original ColoRadar files,
not ColoRadar+ or another radar representation without a compatibility study.
Raw ingestion needs the run's cascade heatmaps/timestamps, LiDAR point clouds and
timestamps, ground-truth poses/timestamps, IMU files and matching sensor calibration
expected by `CascadeDataLoader`. The full `requirements.txt` supplies the retained
research tools; the independent geometry route needs only `requirements-geometry.txt`.

The historical shell dataset launcher remains a directory listing with commented
commands. Use the explicit Python commands in the README. Training retains the
historical `run2` training and `run0`, `run1`, `run4` heldout split. Prepare those
names deliberately, keep sessions disjoint, and retain pairing/calibration records.
Do not interpret a `split` argument in the internal helper as an implemented
random split.

Neither this repository's releases nor the upstream
[learned-depth-imaging-radar releases](https://github.com/rpl-cmu/learned-depth-imaging-radar/releases)
provided a compatible trained checkpoint when checked on 2026-10-29. The upstream
Google Drive sample is a Rerun recording, not a checkpoint or measured radar input.
A legitimate trained run still requires a verified DRNet3D `state_dict`, checkpoint
license/digest and feature contract, matching measured test frames and calibration,
and independent heldout depth/visibility metrics. The inherited illustration and
the newly exported synthetic label geometry do not satisfy those gates. No
real-data accuracy, inference latency, physical radar validation or speedup is claimed.

## Commands used

```bash
python -m pip install -r requirements-geometry.txt
python -m unittest discover -s tests -v
python radardepth.py geometry-demo --output /tmp/radardepth-first --distance 2.5
python radardepth.py geometry-demo --output /tmp/radardepth-changed --distance 3.25
# Optional actual-network CPU loop, separate from measured-data evaluation:
python -m pip install torch==2.15.0
python -m unittest discover -s tests -p test_training_loop.py -v
```

The geometry and changed-input tests compare production results with a separate
scalar nearest-return projector and analytic spherical xyz. No inference engine is
mocked. Raw-ingestion timestamp tests use an original synthetic loader fixture;
they do not verify a downloaded ColoRadar run. Old source and kernel manifests
remain historical records, with explicit source amendments for repaired files.
