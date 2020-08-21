from __future__ import print_function

import os

import numpy as np
import rospkg
import rospy
import torch
import torch.utils.data as data
from sensor_msgs.msg import PointCloud2
from torch import nn

import ros_helper
from configs import ConfigQDH as cfg
from datasets.qdh_inference import QdhInferenceDataset
from models.RandLANet import Network
from utils.data_utils import DataProcessing as DP
from utils.timer import Timer
from utils.data_utils import make_cuda


class InferenceHelper:
    # TODO move hardcoded definitions into config file
    label_to_names = {0: 'unlabeled', 1: 'tripod', 2: 'element'}
    label_to_colors = {0: (255, 255, 255), 1: (255, 0, 0), 2: (0, 255, 0)}

    def __init__(self, checkpoint_path=None):
        # init network
        device = rospy.get_param('/ros_randla_net/network/device')
        if device == 'auto':
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        rospy.loginfo('Use device: {}'.format(self.device.type))
        if self.device.type == 'cpu':
            os.environ['LRU_CACHE_CAPACITY'] = '1'
            rospy.loginfo(
                'Set LRU_CACHE_CAPACITY={} to avoid memory leak'.format(
                    os.getenv('LRU_CACHE_CAPACITY')))

        self.net = Network(cfg)
        self.net.to(self.device)

        # load checkpoint
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            rospy.loginfo("-> loaded checkpoint %s (epoch: %d)" %
                          (checkpoint_path, start_epoch))

        # use multiple gpus if possible
        if self.device.type != 'cpu' and torch.cuda.device_count() > 1:
            rospy.loginfo("Let's use %d GPUs!" % (torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)

        # set model into evaluation mode
        self.net.eval()

        self.debug = rospy.get_param('/ros_randla_net/general/debug')
        self.frame_id = rospy.get_param('/ros_randla_net/general/frame_id')
        self.timer = Timer(enabled=self.debug)

        self.dataset = QdhInferenceDataset(config=cfg,
                                           mode='inference',
                                           pcs=None)
        self.data_loader = None

    def pre_process(self, ros_msg):
        self.timer.restart()
        # convert from ROS PointCloud2 to PCL XYZ
        pcl_xyz = ros_helper.ros_to_pcl(ros_msg, field_type='xyz')
        self.timer.log_and_restart('pre-process: ros_to_pcl')

        # convert to numpy array
        np_xyz = np.asarray(pcl_xyz)

        # # down-sampling
        # pick_idx = np.random.choice(len(np_xyz), 1)
        # selected_idx = QdhDataset.crop_pc(np_xyz, pick_idx)
        # selected_pc = np_xyz[selected_idx, :]
        #
        # # generate fake batch
        # batch_pc = np.expand_dims(selected_pc, axis=0)
        # self.timer.log_and_restart('pre-process: pcl_to_np')
        #
        # # tf amp
        # input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
        # for i in range(cfg.num_layers):
        #     neighbour_idx = DP.knn_batch(batch_pc, batch_pc, cfg.k_n)
        #     sub_points = batch_pc[:, :batch_pc.shape[1] //
        #                           cfg.sub_sampling_ratio[i], :]
        #     pool_i = neighbour_idx[:, :batch_pc.shape[1] //
        #                            cfg.sub_sampling_ratio[i], :]
        #     up_i = DP.knn_batch(sub_points, batch_pc, 1)
        #     input_points.append(batch_pc)
        #     input_neighbors.append(neighbour_idx)
        #     input_pools.append(pool_i)
        #     input_up_samples.append(up_i)
        #     batch_pc = sub_points
        # self.timer.log_and_restart('pre-process: tf_map')
        #
        # # collate
        # inputs = {'xyz': [], 'neigh_idx': [], 'sub_idx': [], 'interp_idx': []}
        # for tmp in input_points:
        #     inputs['xyz'].append(torch.from_numpy(tmp).float())
        # for tmp in input_neighbors:
        #     inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        # for tmp in input_pools:
        #     inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        # for tmp in input_up_samples:
        #     inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        # self.timer.log_and_restart('pre-process: collate')

        # tensor dataset
        self.dataset.set_pcs(np_xyz)

        self.data_loader = data.DataLoader(self.dataset,
                                           batch_size=cfg.inference_batch_size,
                                           shuffle=True,
                                           collate_fn=self.dataset.collate_fn,
                                           num_workers=cfg.num_workers,
                                           drop_last=False)
        batch_data = next(iter(self.data_loader))
        inputs, selected_indices = batch_data

        return pcl_xyz, inputs, selected_indices

    def inference(self, batch_data):
        with torch.no_grad():
            if self.device.type != 'cpu':
                batch_data = make_cuda(batch_data)
            preds = self.net(batch_data)
        return preds

    def post_process(self, ros_msg, pcl_xyz, selected_indices, preds):
        logits = preds
        logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
        logits = logits.max(dim=1)[1].cpu().numpy()
        colors = [self.label_to_colors[0] for _ in range(pcl_xyz.size)]
        for selected_idx in selected_indices:
            for idx, sel_idx in enumerate(selected_idx):
                if logits[idx] != 0:
                    colors[sel_idx] = self.label_to_colors[logits[idx]]
        # colors = [self.label_to_colors[int(label)] for label in logits]
        pcl_xyzrgb = ros_helper.XYZ_to_XYZRGB(pcl_xyz,
                                              color=colors,
                                              use_multiple_colors=True)
        out_msg = ros_helper.pcl_to_ros(pcl_xyzrgb,
                                        frmae_id=self.frame_id,
                                        timestamp=ros_msg.header.stamp)
        return out_msg

    def process(self, ros_msg):
        self.timer.restart()
        pcl_xyz, inputs, selected_indices = self.pre_process(ros_msg)
        preds = self.inference(inputs)
        self.timer.log_and_restart('inference')
        out_msg = self.post_process(ros_msg, pcl_xyz, selected_indices, preds)
        self.timer.log_and_restart('post-process')
        self.timer.print_log()
        return out_msg


class InferenceNode:
    def __init__(self):
        # params
        topic_pcl_sub = rospy.get_param('/ros_randla_net/topics/pcl_sub')
        topic_pcl_pub = rospy.get_param('/ros_randla_net/topics/pcl_pub')
        checkpoint_path = rospy.get_param(
            '/ros_randla_net/network/checkpoint_path')

        # model
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ros_randla_net')
        self.helper = InferenceHelper(
            checkpoint_path=os.path.join(package_path, checkpoint_path))

        # subscription and publishing
        self.pcl_sub = rospy.Subscriber(topic_pcl_sub,
                                        PointCloud2,
                                        self.callback,
                                        queue_size=1)
        self.pcl_pub = rospy.Publisher(topic_pcl_pub,
                                       PointCloud2,
                                       queue_size=1)

    def callback(self, ros_msg):
        out_msg = self.helper.process(ros_msg)
        self.pcl_pub.publish(out_msg)
