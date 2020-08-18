import numpy as np
import pcl
import torch
import torch.utils.data as data
from sklearn.neighbors import KDTree

from configs import ConfigQDH as cfg
from utils.data_utils import DataProcessing as DP


class QdhDataset(data.Dataset):
    def __init__(self, cfg, mode, test_id=None):
        self.name = 'qdh'
        self.label_to_names = {0: 'background', 1: 'triangle'}
        self.num_classes = cfg.num_classes
        # self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        # self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        # self.ignored_labels = np.sort([0])

        # if mode == 'test':
        #     self.test_scan_number = str(test_id)

        self.mode = mode
        train_list, val_list, test_list = DP.get_file_list(cfg)
        if mode == 'train':
            self.data_list = train_list
        # elif mode == 'eval':
        #     self.data_list = val_list
        elif mode == 'test':
            self.data_list = test_list

        # self.data_list = self.data_list[0]
        # self.data_list = DP.shuffle_list(self.data_list)

        # self.possibility = []
        # self.min_possibility = []
        # if mode == 'test':
        #     path_list = self.data_list
        #     for test_file_name in path_list:
        #         points = np.load(test_file_name)
        #         self.possibility += [np.random.rand(points.shape[0]) * 1e-3]
        #         self.min_possibility += [float(np.min(self.possibility[-1]))]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        short_name, selected_pc, selected_labels, selected_idx, masks = self.spatially_regular_gen(
            item)
        return short_name, selected_pc, selected_labels, selected_idx, masks

    def spatially_regular_gen(self, item):
        # Generator loop
        # if self.mode != 'test':
        cloud_ind = item
        pc_path = self.data_list[cloud_ind]
        short_name, coords, labels = self.get_data(pc_path)
        # crop a small point cloud
        if self.mode == 'train':
            pick_idx = np.random.choice(len(coords), 1)
            selected_idx = self.crop_pc(coords, pick_idx)
        else:
            selected_idx = np.arange(len(coords))
            selected_idx = DP.shuffle_idx(selected_idx)
        selected_pc = coords[selected_idx, :]
        if self.mode != 'test':
            selected_labels = labels[selected_idx, :].astype('int64')
            # masks: not for instance but for semantic label
            masks = np.zeros((selected_labels.shape[0], cfg.num_classes),
                             dtype=np.float32)
            masks[np.arange(selected_labels.shape[0]), selected_labels[:,
                                                                       0]] = 1
        else:
            # selected_labels = []
            # masks = 0
            selected_labels = labels[selected_idx, :].astype('int64')
            # masks: not for instance but for semantic label
            masks = np.zeros((selected_labels.shape[0], cfg.num_classes),
                             dtype=np.float32)
            masks[np.arange(selected_labels.shape[0]), selected_labels[:,
                                                                       0]] = 1

        # print(selected_idx.shape)
        return short_name, selected_pc, selected_labels, selected_idx, masks

    def get_data(self, fname):
        short_name = fname.split('/')[-1]
        if self.mode == 'test':
            scan = pcl.load_XYZI(fname)
            xyzi = []
            for i in range(scan.width):
                xyzi.append(scan.__getitem__(i))
            xyzi = np.array(xyzi)
            coords = xyzi[:, :3]
            labels = xyzi[:, 3]
            labels = labels.reshape(-1, 1)
        else:
            scan = pcl.load_XYZI(fname)
            xyzi = []
            for i in range(scan.width):
                xyzi.append(scan.__getitem__(i))
            xyzi = np.array(xyzi)
            coords = xyzi[:, :3]
            labels = xyzi[:, 3]
            labels = labels.reshape(-1, 1)

        return short_name, coords, labels

    @staticmethod
    def crop_pc(coords, pick_idx):
        # crop a fixed size point cloud for training
        center_point = coords[pick_idx, :].reshape(1, -1)
        search_tree = KDTree(coords)
        selected_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        selected_idx = DP.shuffle_idx(selected_idx)
        return selected_idx

    # @torchsnooper.snoop()
    def tf_map(self, batch_pc, batch_label, batch_sel_idx, batch_masks):
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        # features = np.concatenate((batch_pc, batch_intensity.reshape(batch_intensity.shape[0], -1, 1)), axis=2)
        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_batch(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] //
                                  cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] //
                                   cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_batch(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_pc, batch_label, batch_sel_idx, batch_masks]

        return input_list

    # @torchsnooper.snoop()
    def collate_fn(self, batch):
        short_name, coords, labels, selected_idx, masks = [], [], [], [], []
        for i in range(len(batch)):
            short_name.append(batch[i][0])
            coords.append(batch[i][1])  # pointcloud
            labels.append(batch[i][2])  # index
            selected_idx.append(batch[i][3])  # frame
            masks.append(batch[i][4])  # masks
        coords = np.stack(coords)
        labels = np.stack(labels)
        selected_idx = np.stack(selected_idx)
        masks = np.stack(masks)

        flat_inputs = self.tf_map(coords, labels, selected_idx, masks)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers:2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers +
                                                        1].astype(np.int64))
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers +
                                                            2])
        inputs['masks'] = torch.from_numpy(flat_inputs[4 * num_layers + 3])

        return short_name, inputs
