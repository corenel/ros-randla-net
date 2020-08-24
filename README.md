# ros-randla-net
ROS node for RandLA-Net to achieve point cloud segmentation.

## Environment

- Ubuntu 16.04 (Recommended) / 18.04
- CUDA 10.0/10.1
- ROS Kinetic / Melodic
- Python 2 (running as ROS node) / Python 3 (training)

> Three exists something wrong with `libpcl-dev` (1.8.1) in Ubuntu 18.04 with `python-pcl` (0.3.0a1), which may cause a  `libpcl_keypoints version mismatch` error. Check [this issue](https://github.com/strawlab/python-pcl/issues/296) for a temporary solution before the author update this package.


## Installation

1. Clone this repo into your workspace

2. Install dependencies

   - Use system Python 2

     ```bash
     $ python2 -m pip install -U pip
     $ cd /path/to/this/repo
     $ python2 -m pip install Cython
     $ python2 -m pip install -r requirements.txt
     ```

   - Use conda environment

     ```bash
     $ conda create -n py27 python=2.7
     $ conda activate py27
     $ python2 -m pip install -U pip
     $ cd /path/to/this/repo
     $ python2 -m pip install Cython
     $ python2 -m pip install -r requirements.txt
     $ python2 -m pip install rospkg catkin-pkg catkin-tools empy
     ```

   - Use offline installation

      ```bash
      # on computer with Internet access
      $ mkdir -p ~/Downloads/wheels
      $ python -m pip install -U pip --user
      $ python -m pip download pip -d ~/Downloads/wheels
      $ python -m pip download -r requirements.txt -d ~/Downloads/wheels
      $ python -m pip download Cython rospkg catkin-pkg catkin-tools empy -d ~/Downloads/wheels
      # copy wheels to computer without Internet access
      # and run the following commands to install packages offline
      $ python -m pip install -U pip --no-index --find-links=/path/to/wheels --user
      $ python -m pip install -U Cython --no-index --find-links=/path/to/wheels --user
      $ python -m pip install -U -r requirements.txt --no-index --find-links=/path/to/wheels --user
      $ python -m pip install -U rospkg catkin-pkg catkin-tools empy --no-index --find-links=/path/to/wheels --user
      ```

3. Build

   ```bash
   # build extensions for this repo
   # use 'build_deps_py3.sh' if you're using Python 3 for training
   $ cd /path/to/this/repo
   $ ./script/build_deps.sh
   # build the whole workspace
   $ cd /path/to/workspace/root
   $ catkin_make
   # source setup.bash/setup.zsh or whatever
   $ source devel/setup.bash
   ```

## Usage

### Prepare checkpoints and models

#### Use pre-trained weights

1. Download `assets.tar.gz` from the Release page
2. Extract `assets.tar.gz` and put files into `assets` directory

### Launch node

1. Modify the `config/params.yml`
2. Run one of provided launch files:
   - ` pointcloud_segmentation.launch`: subscribe point cloud and do segmentation

### Topics

- `/source_cloud` (`sensor_msgs/PointCloud2`): Input point cloud (in `PointXYZ` format)

- `/source_cloud_segmented` (`sensor_msgs/PointCloud2`): Segmented point cloud (in `PointXYZRGB` format)

  - The RGB value of ``PointXYZRGB`` in `sensor_msg/PointField` indicates the point label

    | Label | Name      | RGB             | Notes        |
    | ----- | --------- | --------------- | ------------ |
    | 0     | Unlabeled | `(255,255,255)` |              |
    | 1     | Tripod    | `(255,0,0)`     |              |
    | 2     | Element   | `(0,255,0)`     | Not used yet |

## Performance

- `num_points=4096`

### Speed

| Hardware           | Throughout |
| ------------------ | ---------- |
| GPU (RTX 2070 8GB) | 4.8Hz      |
| CPU (i5-9400)      | 1.2Hz      |
| CPU (i5-8250U)     | 0.4Hz      |

### Profiling

On GPU (RTX 2070 8GB):

| Item                    | Time (ms) |
| ----------------------- | --------- |
| pre-process: ros_to_pcl | 26.6530   |
| pre-process: pcl_to_np  | 0.1932    |
| pre-process: tf_map     | 86.7923   |
| pre-process: collate    | 0.7529    |
| inference               | 10.0480   |
| post-process            | 79.7921   |
| **Total**               | 204.2314  |

## Training

1. Modify `configs/qdh.py`

   - `root`: root path to train set
   - `test_root`: root path to test se
   - `num_points`: number of points fed into network
   - `num_classes`: number of classes
   - `label_to_names`: mapping from class indices into class names
   - `batch_size`: batch size for training
   - `eval_batch_size`: batch size for evaluation
   - `inference_batch_size`: batch size for inference in ROS node
   - `crop_pc_on_inference`: whether or not crop the number point clouds to `num_points` in inference process
   - `max_epoch`: maximum number of training epochs
   - `use_full_set_pc_in_training`: whether or not to use full set point cloud (instead of `num_points`) in training process
   - `ignore_bg_labels_in_training`: whether or not to ignore background label (`0`) for loss calculation in training
   - `optimizer`: type of optimizer (`adam` or `lookahead`)
   - `use_data_augmentation`: whether or not to use data augmentation (random rotation/translation) in training process
   - `no_norm`: whether or not to use point cloud normalization in training process

2. Get to training environment and run:

   ```bash
   $ python3 train.py
   ```

3. Test model for single PCD file:

   ```bash
   $ python3 test.py --input {path/to/pcd/file} --checkpoint {path/to/checkpoint/file}
   ```


## TODO

- [ ] Accelerate `tf_map` step in pre-process procedure
- [x] Add code for training models
- [x] Support `element` class

## Known issues

- In the realworld scene, there is a confounding phenomenon between the two categories of `tripod` and `element`, especially when the object is close to the lidar.