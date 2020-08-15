# ros-randla-net
ROS node for RandLA-Net to achieve point cloud segmentation.


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
      $ mkdir ~/Downloads/wheels
      $ python -m pip install -U pip --user
      $ python -m pip download pip -d ~/Downloads/wheels
      $ python -m pip download -r requirements.txt -d ~/Downloads/wheels
      $ python -m pip download rospkg catkin-pkg catkin-tools empy -d ~/Downloads/wheels
      # copy wheels to computer without Internet access
      # and run the following commands to install packages offline
      $ python -m pip install -U pip --no-index --find-links=/path/to/wheels --user
      $ python -m pip install Cython --no-index --find-links=/path/to/wheels --user
      $ python -m pip install -r requirements.txt --no-index --find-links=/path/to/wheels --user
      $ python -m pip install -U rospkg catkin-pkg catkin-tools empy --no-index --find-links=/path/to/wheels --user
      ```

3. Build

   ```bash
   # build extensions for this repo
   $ cd /path/to/this/repo
   $ ./script/build_ops.sh
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

### Speed

| Hardware           | Throughout |
| ------------------ | ---------- |
| GPU (RTX 2070 8GB) | 4.8Hz      |
| CPU (i5-9400)      | 1.2Hz      |

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

