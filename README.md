# RadarDepth

**From imaging radar to cylindrical depth maps and indoor geometry.**

Learn depth from ColoRadar measurements with LiDAR supervision. A learned visibility mask removes directions the radar cannot observe before the depth map is projected into geometry.

<a id="pipeline"></a>

<img src="assets/image.png" width="400" alt="Retained research illustration: two depth-image strips above an indoor radar and LiDAR point-cloud comparison, with radar in orange and LiDAR in blue.">

*Orange denotes radar geometry; blue denotes LiDAR. This retained qualitative comparison is not a new evaluation run or an accuracy measurement.*

[Inspect without a dataset](#start-without-a-dataset) · [Prepare and run](#run-the-research-pipeline) · [Network source](networks/DepthRefinement/model.py) · [Research reference](docs/REFERENCE.md)

<a id="verification"></a>

## Start without a dataset

The NumPy-only route computes new conversion labels and geometry, without a model,
viewer, downloaded dataset or GPU:

```bash
python -m pip install -r requirements-geometry.txt
python radardepth.py geometry-demo --output /tmp/radardepth-first --distance 2.5
python radardepth.py geometry-demo --output /tmp/radardepth-changed --distance 3.25
python -m unittest discover -s tests -v
```

Each demo writes an original ideal radar/LiDAR fixture, converted radar and depth
arrays, `geometry.ply`, and a hashed `result.json`. The nearer LiDAR return wins;
rear points are excluded. The recovered point is compared with an analytic
spherical-coordinate reference. Use new output directories to preserve prior
results. **This is synthetic label conversion and geometry, not learned depth
estimation or a real radar measurement.**

[Reproduction contracts and remaining gates](docs/REPRODUCIBILITY.md) distinguish
this working numerical subset, CPU network training tests, and unavailable trained
inference. With only NumPy installed, two optional Torch tests are explicitly
skipped. The separate CPU CI job runs those tests using the actual DRNet3D network.

<a id="set-up"></a>

## Run the research pipeline

The full route needs **ColoRadar runs, the matching calibration and model weights**. Measurements and trained checkpoints are not bundled. Use a separate Python environment for the application dependencies:

```bash
python -m pip install -r requirements.txt matplotlib
```

The retained [dataset guide](docs/REFERENCE.md#dataset) explains the layouts, but its shell launcher currently only lists directories: the conversion commands inside it are commented out. Use the two Python conversion scripts explicitly. From the repository root, for a bounded first run:

```bash
python generate_dataset/raw_dataset_to_numpy.py -s PATH_TO_COLORADAR/RUN -c PATH_TO_COLORADAR/calib -o PATH_TO_TEMPORARY/run0 --max-frames 10
python radardepth.py convert -l PATH_TO_TEMPORARY/run0/raw/lidar -r PATH_TO_TEMPORARY/run0/raw/radar -m PATH_TO_TEMPORARY/run0/calib/radarMapping.npz -c PATH_TO_TEMPORARY/run0/calib/calibration.npz -o PATH_TO_TRAINING_DATA/run0 --max-frames 10
```

Use absolute paths for the `PATH_TO_...` placeholders, and repeat with the appropriate output run names. Training currently selects `run2` and tests on `run0`, `run1` and `run4` in [`train.py`](train.py), so prepare those runs consistently. Raw ColoRadar ingestion has not been executed on measured data here; the same NumPy conversion and geometry core has been exercised on the ideal fixture above. Use each run's generated calibration rather than assuming the bundled rig matches another capture. Training and evaluation retain the following entry points:

```bash
python radardepth.py train -d PATH_TO_TRAINING_DATA -o output -s true
python radardepth.py evaluate -m MODEL_PATH -c calib/radarMapping.npz -r calib/calibration.npz -d PATH_TO_TRAINING_DATA/run0 -l PATH_TO_TEMPORARY/run0/raw/lidar/
```

`-s true` enables the segmentation branch and is required by the retained training parser. `train` forwards to [`train.py`](train.py); `evaluate` forwards to [`eval.py`](eval.py). Evaluation opens a Rerun viewer comparing predicted radar geometry, LiDAR labels and raw LiDAR points. The full measured-data and trained-checkpoint route has not been reproduced. The training loop is tested separately with original synthetic tensors; this does not establish learned accuracy.

| To inspect | Open |
| --- | --- |
| Radar-volume network, depth regression and visibility branch | [`model.py`](networks/DepthRefinement/model.py) |
| Raw captures → NumPy → training layout | [Dataset preparation](docs/REFERENCE.md#dataset) |
| Dataset format and paired supervision | [`dataloader/dataset.py`](dataloader/dataset.py) |
| Calibration and geometry comparison | [`eval.py`](eval.py) |
| Passed checks and unavailable checks | [Verification record](docs/VERIFICATION.md) |

If help works but training fails on an import, help has only verified the dependency-free dispatcher. Install the full application requirements in the same environment. Missing data or checkpoint errors require the external artifacts above; the retained illustration is not a bundled model output to replay.

<a id="source"></a>

## License

The [project license](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) cover the research source and bundled ColoRadar tools. [Source provenance](NOTICE.md).
