import numpy as np
import torch
import torch.utils.data as data
from sklearn.neighbors import KDTree

from utils.data_utils import DataProcessing as DP


class BaseDataset(data.Dataset):
    def __init__(self, config, mode):
        self.cfg = config
        self.mode = mode

    def data_aug(self, pc):
        if self.cfg.use_data_augmentation:
            theta = self.cfg.rotation_jitter
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),
                                         np.cos(theta)]])
            position_offsets = np.random.normal(0,
                                                self.cfg.position_jitter,
                                                size=pc.shape)
            displacement = np.random.normal(0,
                                            self.cfg.displacement,
                                            size=(1, 3))
            pc[:, [0, 2]] = pc[:, [0, 2]].dot(rotation_matrix)
            pc += position_offsets
            pc += displacement

        return pc

    def normalize(self, pc):
        if not self.cfg.no_norm:
            center = np.expand_dims(np.mean(pc, axis=0), 0)
            pc = pc - center
            dist = np.max(np.sqrt(np.sum(pc**2, axis=1)), 0)
            pc = pc / dist  # scale
        else:
            center = np.zeros((1, 3))
            dist = 1.0
        return pc, center, dist

    def crop_pc(self, coords, pick_idx):
        # crop a fixed size point cloud for training
        center_point = coords[pick_idx, :].reshape(1, -1)
        search_tree = KDTree(coords)
        selected_idx = search_tree.query(center_point,
                                         k=self.cfg.num_points)[1][0]
        selected_idx = DP.shuffle_idx(selected_idx)
        return selected_idx

    def tf_map(self,
               batch_pc,
               batch_label=None,
               batch_sel_idx=None,
               batch_masks=None):
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        for i in range(self.cfg.num_layers):
            neighbour_idx = DP.knn_batch(batch_pc, batch_pc, self.cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] //
                                  self.cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] //
                                   self.cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_batch(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        inputs = {'xyz': []}
        for tmp in input_points:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in input_neighbors:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in input_pools:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in input_up_samples:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        if batch_label is not None:
            inputs['labels'] = torch.from_numpy(batch_label.astype(np.int64))
        if batch_sel_idx is not None:
            inputs['input_inds'] = torch.from_numpy(batch_sel_idx)
        if batch_masks is not None:
            inputs['masks'] = torch.from_numpy(batch_masks)

        return inputs
