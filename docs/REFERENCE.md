# Learned Depth Estimation of 3D Imaging Radar for Indoor Mapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

3D imaging radar offers robust perception capability through
visually demanding environments due to the
unique penetrative and reflective properties of millimeter waves
(mmWave). Current approaches for 3D perception with imaging
radar require knowledge of environment geometry, accumulation
of data from multiple frames for perception, or access to
between-frame motion. Imaging radar presents an additional
difficulty due to the complexity of its data representation. To
address these issues, and make imaging radar easier to use for
downstream robotics tasks, we propose a learning-based method
that regresses radar measurements into cylindrical depth maps
using LiDAR supervision. Due to the limitation of the regression
formulation, directions where the radar beam could not reach
will still generate a valid depth. To address this issue, our
method additionally learns a 3D filter to remove those pixels.
Experiments show that our system generates visually accurate
depth estimation. Furthermore, we confirm the overall ability
to generalize in the indoor scene using the estimated depth for
probabilistic occupancy mapping with ground truth trajectory.

[**IEEEXplore**](https://ieeexplore.ieee.org/abstract/document/9981572)

<img src="https://github.com/rpl-cmu/learned-depth-imaging-radar/blob/main/assets/image.png" width = 55% height = 55%/>

## About
Code Repository for IROS 2022 Submission __Learned Depth Estimation of 3D Imaging Radar for Indoor Mapping__

If you find this project useful for your research, please cite:
```
@INPROCEEDINGS{9981572,
  author={Xu, Ruoyang and Dong, Wei and Sharma, Akash and Kaess, Michael},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Learned Depth Estimation of 3D Imaging Radar for Indoor Mapping}, 
  year={2022},
  volume={},
  number={},
  pages={13260-13267},
  keywords={Three-dimensional displays;Radar measurements;Navigation;Imaging;Radar;Radar imaging;Millimeter wave radar},
  doi={10.1109/IROS47612.2022.9981572}}
```

## How to Use
### Installation
A list of dependencies are provided in the `requirements.txt`.

### Dataset
We trained and tested our dataset on the ColoRadar dataset. To use it with this code, please structure and unzip the dataset in the following format.
Not all runs from the following are needed, but please include at least 4.
```zsh
└── ColoRadar
    ├── 12_21_2020_arpg_lab_run0
    ├── 12_21_2020_arpg_lab_run1
    ├── 12_21_2020_arpg_lab_run2
    ├── 12_21_2020_arpg_lab_run3
    ├── 12_21_2020_arpg_lab_run4
    ├── 12_21_2020_ec_hallways_run0
    ├── 12_21_2020_ec_hallways_run1
    ├── 12_21_2020_ec_hallways_run2
    ├── 12_21_2020_ec_hallways_run4
    └── calib
```
Then provide a directory for storing a numpy-fied version of the dataset, and a separate directory dataset to be used by DataLoader. This directory could look like the following. This approach is purely due to legacy development reason and there is no plan to change this.
```zsh
└── converted_dataset
    ├── temp
    └── training_data
```

We can then generate this dataset to the format we could directly use. 
```zsh
learned-depth-imaging-radar $ cd generate_dataset
learned-depth-imaging-radar/generate_dataset $ ./dataset_generation_script.zsh PATH_TO_COLORADAR PATH_TO_COLORADAR_CALIB_FOLDER PATH_TO_TEMPORARY PATH_TO_TRAINING
# using above as an example:
learned-depth-imaging-radar/generate_dataset $ ./dataset_generation_script.zsh ColoRadar ColoRadar/calib converted_dataset/temp converted_dataset/training_data
```
This will create a directory roughly looks like the following, at which point `training_data` can be used and `temp` can be deleted.
```zsh
.
├── temp
│   └── run0
│       ├── 2D
│       ├── base_pose
│       ├── calib
│       ├── depthImg
│       └── raw
└── training_data
    └── run0
        ├── lidar
        └── radar
```

### Training
Create a directory at `output` for storing intermediate checkpoints
```zsh
$ python train.py -d PATH_TO_TRAINING_DATA -o ./output
```

### Evaluation
We provide a script to quickly compare the output depth image with label and the raw LiDAR measurements in point cloud form. The following is an example.
```zsh
$ python eval.py -m MODEL_PATH -c calib/radarMapping.npz -r calib/calibration.npz -d PATH_TO_TRAINING/run0 -l PATH_TO_TEMPORARY/run0/raw/lidar/
```
This script will open a rerun window where the the pointclouds will be overlaid together for comparison.
We provide a sample recording as this link: [Google Drive](https://drive.google.com/file/d/1oW0nKFtIk8ROyMUySTyVFfU3c23E0oi3/view?usp=sharing) Where you can open it with 
```zsh
$ rerun sample_recording.rrd
```
