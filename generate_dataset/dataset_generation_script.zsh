#!/bin/zsh
dataset_dir=$1
calib_dir=$2
output_dir=$3
training_set_dir=$4

# This detour is required
ds_idx=start
start=0
for file in ${dataset_dir}/*(/)
do
    if [[ "$file" == */calib ]]; then
        continue
    fi
    echo $file
    # python raw_dataset_to_numpy.py -s $file -c $calib_dir -o "${output_dir}/run${start}"
    # python numpy_to_trainset.py \
    #     --lidar_sequence "${output_dir}/run${start}/raw/lidar/" \
    #     --radar_sequence "${output_dir}/run${start}/raw/radar/" \
    #     --radar_calib_npz ../calib/radarMapping.npz \
    #     --rig_calib_npz ../calib/calibration.npz \
    #     --output "${training_set_dir}/run${start}" \
    #     -lower 8 -upper 24
    # (($ds_idx++))
done
