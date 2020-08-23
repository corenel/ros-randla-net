import ctypes
import glob
import os
import struct
import sys
from random import randint

import numpy as np
import open3d
import pandas as pd
import pcl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# from utils.cpp_wrappers.cpp_subsampling import grid_subsampling as cpp_subsampling
from utils.nearest_neighbors.lib.python import nearest_neighbors


class DataProcessing:
    def __init__(self):
        pass

    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename,
                            header=None,
                            delim_whitespace=True,
                            dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename,
                               header=None,
                               delim_whitespace=True,
                               dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_file_list(cfg):
        # seq_list = np.sort(os.listdir(cfg.root))

        train_file_list, val_file_list, test_file_list = [], [], []
        train_file_list = sorted(glob.glob(os.path.join(cfg.root, '*.pcd')))
        # for sq in cfg.valid:
        # # for sq in range(11):
        #     seqstr = '{0:02d}'.format(int(sq))
        #     val_file_list += sorted(glob.glob(os.path.join(cfg.root, seqstr, 'velodyne', '*.bin')))
        # # for sq in cfg.valid:     # FIXME
        # # for sq in ['00','01','02','03','04','05','06','07','08','09','10']:
        # for sq in ['21']: # '11','12','13','14','15','16','17','18','19','20'
        #     seqstr = '{0:02d}'.format(int(sq))
        # test_file_list = sorted(glob.glob(os.path.join('/media/kx/yangxm/qdh/data/ROI_scan', '*.pcd')))
        test_file_list = sorted(glob.glob(os.path.join(cfg.test_root,
                                                       '*.pcd')))

        if os.path.exists(os.path.join(cfg.root, 'invalid_files.txt')):
            with open(os.path.join(cfg.root, 'invalid_files.txt')) as f:
                invalid_file_list = [line.strip() for line in f.readlines()]
        else:
            invalid_file_list = []

        # train_file_list = train_file_list[:22]
        # val_file_list = val_file_list[:22]

        return train_file_list, val_file_list, test_file_list, invalid_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """
        neighbor_idx = nearest_neighbors.knn_batch(support_pts,
                                                   query_pts,
                                                   k,
                                                   omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def knn_batch(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts,
                                                   query_pts,
                                                   k,
                                                   omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    # @staticmethod
    # def grid_sub_sampling(points,
    #                       features=None,
    #                       labels=None,
    #                       grid_size=0.1,
    #                       verbose=0):
    #     """
    #     CPP wrapper for a grid sub_sampling (method = barycenter for points and features
    #     :param points: (N, 3) matrix of input points
    #     :param features: optional (N, d) matrix of features (floating number)
    #     :param labels: optional (N,) matrix of integer labels
    #     :param grid_size: parameter defining the size of grid voxels
    #     :param verbose: 1 to display
    #     :return: sub_sampled points, with features and/or labels depending of the input
    #     """
    #
    #     if (features is None) and (labels is None):
    #         return cpp_subsampling.compute(points,
    #                                        sampleDl=grid_size,
    #                                        verbose=verbose)
    #     elif labels is None:
    #         return cpp_subsampling.compute(points,
    #                                        features=features,
    #                                        sampleDl=grid_size,
    #                                        verbose=verbose)
    #     elif features is None:
    #         return cpp_subsampling.compute(points,
    #                                        classes=labels,
    #                                        sampleDl=grid_size,
    #                                        verbose=verbose)
    #     else:
    #         return cpp_subsampling.compute(points,
    #                                        features=features,
    #                                        classes=labels,
    #                                        sampleDl=grid_size,
    #                                        verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([
                3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                650464, 791496, 88727, 1284130, 229758, 2272837
            ],
                                     dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array([
                5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860,
                269353
            ],
                                     dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([
                55437630, 320797, 541736, 2578735, 3274484, 552662, 184064,
                78858, 240942562, 17294618, 170599734, 6369672, 230413074,
                101130274, 476491114, 9833174, 129609852, 4506626, 1168181
            ])
        elif dataset_name is 'qdh':
            # TODO check actual number
            num_per_class = np.array([78776979, 15241283, 23334316],
                                     dtype=np.int32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)


def make_cuda(batch_data):
    for key in batch_data:
        if type(batch_data[key]) is list:
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].cuda()
        else:
            batch_data[key] = batch_data[key].cuda()
    return batch_data


# def visualize(xyz, pred, short_name):
#     color = [[245 / 255, 150 / 255, 100 / 255],
#              [90 / 255, 30 / 255, 150 / 255], [80 / 255, 240 / 255, 150 / 255]]
#     xyz = xyz.squeeze().cpu().detach().numpy()
#     pred = pred.cpu().detach().numpy()
#
#     pred_label_set = list(set(pred))
#     pred_label_set.sort()
#     print(pred_label_set)
#     viz_point = open3d.geometry.PointCloud()
#     point_cloud = open3d.geometry.PointCloud()
#     for id_i, label_i in enumerate(pred_label_set):
#         index = np.argwhere(pred == label_i).reshape(-1)
#         sem_cluster = xyz[index, :]
#         point_cloud.points = open3d.utility.Vector3dVector(sem_cluster)
#         point_cloud.paint_uniform_color(color[id_i])
#         viz_point += point_cloud
#
#     # open3d.visualization.draw_geometries([viz_point],
#     #                                      window_name=short_name[0],
#     #                                      width=1920,
#     #                                      height=1080,
#     #                                      left=50,
#     #                                      top=50)
#
#     vis = open3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(viz_point)
#     vis.update_geometry(viz_point)
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(
#         os.path.join('results', os.path.basename(short_name)))
#     vis.destroy_window()


def random_color_gen():
    """
    Generates a random color

    :return: 3 elements, R, G, and B
    :rtype: list
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]


def XYZRGB_to_XYZ(XYZRGB_cloud):
    """
    Converts a PCL XYZRGB point cloud to an XYZ point cloud (removes color info)

    :param XYZRGB_cloud: A PCL XYZRGB point cloud
    :type XYZRGB_cloud: PointCloud_PointXYZRGB
    :return: A PCL XYZ point cloud
    :rtype: PointCloud_PointXYZ
    """
    XYZ_cloud = pcl.PointCloud()
    points_list = []

    for data in XYZRGB_cloud:
        points_list.append([data[0], data[1], data[2]])

    XYZ_cloud.from_list(points_list)
    return XYZ_cloud


def XYZ_to_XYZRGB(XYZ_cloud, color, use_multiple_colors=False):
    """
    Converts a PCL XYZ point cloud to a PCL XYZRGB point cloud

    All returned points in the XYZRGB cloud will be the color indicated
    by the color parameter.

    :param XYZ_cloud: A PCL XYZ point cloud
    :type XYZ_cloud: PointCloud_XYZ
    :param color: 3-element list of integers [0-255,0-255,0-255]
    :type color: list
    :param use_multiple_colors: use more than one color
    :type use_multiple_colors: bool
    :return: A PCL XYZRGB point cloud
    :rtype: PointCloud_PointXYZRGB
    """
    XYZRGB_cloud = pcl.PointCloud_PointXYZRGB()
    points_list = []

    float_rgb = rgb_to_float(color) if not use_multiple_colors else None

    for idx, data in enumerate(XYZ_cloud):
        float_rgb = rgb_to_float(
            color[idx]) if use_multiple_colors else float_rgb
        points_list.append([data[0], data[1], data[2], float_rgb])

    XYZRGB_cloud.from_list(points_list)
    return XYZRGB_cloud


def XYZ_to_XYZI(XYZ_cloud, color, use_multiple_colors=False):
    XYZI_cloud = pcl.PointCloud_PointXYZI()
    points_list = []

    for idx, data in enumerate(XYZ_cloud):
        intensity = int(color[idx]) if use_multiple_colors else int(color)
        points_list.append([data[0], data[1], data[2], intensity])

    XYZI_cloud.from_list(points_list)
    return XYZI_cloud


def rgb_to_float(color):
    """
    Converts an RGB list to the packed float format used by PCL

    From the PCL docs:
    "Due to historical reasons (PCL was first developed as a ROS package),
     the RGB information is packed into an integer and casted to a float"

    :param color: 3-element list of integers [0-255,0-255,0-255]
    :type color: list
    :return: RGB value packed as a float
    :rtype: float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb


def float_to_rgb(float_rgb):
    """
    Converts a packed float RGB format to an RGB list

    :param float_rgb: RGB value packed as a float
    :type float_rgb: float
    :return: 3-element list of integers [0-255,0-255,0-255]
    :rtype: list
    """
    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    color = [r, g, b]

    return color


label_to_names = {0: 'unlabeled', 1: 'tripod', 2: 'element'}
label_to_colors = {0: (255, 255, 255), 1: (255, 0, 0), 2: (0, 255, 0)}


def visualize(preds, pcs_xyz, selected_indices, outpath):
    colors = [label_to_colors[0] for _ in range(pcs_xyz.size)]
    for selected_idx in selected_indices:
        for idx, sel_idx in enumerate(selected_idx):
            if int(preds[idx]) != 0:
                colors[sel_idx] = label_to_colors[int(preds[idx])]
    pcl_xyzrgb = XYZ_to_XYZRGB(pcs_xyz, color=colors, use_multiple_colors=True)
    pcl.save(pcl_xyzrgb, outpath.replace('.pcd', '.ply'))
