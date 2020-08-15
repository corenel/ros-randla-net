from __future__ import print_function

import os

import numpy as np
import rospkg
import rospy
import torch
from sensor_msgs.msg import PointCloud2
from torch import nn

import ros_helper
from RandLANet import Network
from helper_tool import ConfigQDH as cfg
from helper_tool import DataProcessing as DP
from utils.timer import Timer


class InferenceHelper:
    # TODO move hardcoded definitions into config file
    label_to_names = {0: 'unlabeled',
                      1: 'triangle',
                      2: 'element'}
    label_to_colors = {0: (255, 255, 255),
                       1: (255, 0, 0),
                       2: (0, 255, 0)}

    def __init__(self, checkpoint_path=None):
        # init network
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Network(cfg)
        self.net.to(self.device)

        # load checkpoint
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            rospy.loginfo("-> loaded checkpoint %s (epoch: %d)" %
                          (checkpoint_path, start_epoch))

        # use multiple gpus if possible
        if torch.cuda.device_count() > 1:
            rospy.loginfo("Let's use %d GPUs!" % (torch.cuda.device_count()))
            net = nn.DataParallel(self.net)

        # set model into evaluation mode
        self.net.eval()

        self.debug = rospy.get_param('/ros_randla_net/general/debug')
        self.timer = Timer(enabled=self.debug)

    def pre_process(self, ros_msg):
        self.timer.restart()
        # convert from ROS PointCloud2 to PCL XYZRGB
        pcl_xyz = ros_helper.ros_to_pcl(ros_msg, field_type='xyz')
        self.timer.log_and_restart('pre-process: ros_to_pcl')

        # convert to numpy array
        np_xyz = np.asarray(pcl_xyz)
        batch_pc = np.stack([np_xyz])
        self.timer.log_and_restart('pre-process: pcl_to_np')

        # tf amp
        input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points
        self.timer.log_and_restart('pre-process: tf_map')

        # collate
        inputs = {'xyz': [], 'neigh_idx': [], 'sub_idx': [], 'interp_idx': []}
        for tmp in input_points:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        for tmp in input_neighbors:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        for tmp in input_pools:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        for tmp in input_up_samples:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        self.timer.log_and_restart('pre-process: collate')

        return inputs, pcl_xyz

    def inference(self, batch_data):
        with torch.no_grad():
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()
            preds = self.net(batch_data)
        return preds

    def post_process(self, pcl_xyz, preds):
        logits = preds
        logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
        logits = logits.max(dim=1)[1].cpu().numpy()
        colors = [self.label_to_colors[int(label)] for label in logits]
        pcl_xyzrgb = ros_helper.XYZ_to_XYZRGB(pcl_xyz, color=colors, use_multiple_colors=True)
        out_msg = ros_helper.pcl_to_ros(pcl_xyzrgb)
        return out_msg

    def process(self, ros_msg):
        self.timer.restart()
        inputs, pcl_xyz = self.pre_process(ros_msg)
        preds = self.inference(inputs)
        self.timer.log_and_restart('inference')
        out_msg = self.post_process(pcl_xyz, preds)
        self.timer.log_and_restart('post-process')
        self.timer.print_log()
        return out_msg


class InferenceNode:
    def __init__(self):
        # params
        topic_pcl_sub = rospy.get_param('/ros_randla_net/topics/pcl_sub')
        topic_pcl_pub = rospy.get_param('/ros_randla_net/topics/pcl_pub')
        checkpoint_path = rospy.get_param('/ros_randla_net/network/checkpoint_path')

        # model
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ros_randla_net')
        self.helper = InferenceHelper(
            checkpoint_path=os.path.join(package_path, checkpoint_path))

        # subscription and publishing
        self.pcl_sub = rospy.Subscriber(topic_pcl_sub, PointCloud2, self.callback, queue_size=3)
        self.pcl_pub = rospy.Publisher(topic_pcl_pub, PointCloud2, queue_size=3)

    def callback(self, ros_msg):
        out_msg = self.helper.process(ros_msg)
        self.pcl_pub.publish(out_msg)
