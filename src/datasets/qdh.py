import numpy as np
import pcl

from utils.data_utils import DataProcessing as DP
from .base import BaseDataset


class QdhDataset(BaseDataset):
    def __init__(self, config, mode):
        super(QdhDataset, self).__init__(config, mode)
        self.name = 'qdh'

        train_list, val_list, test_list, invalid_file_list = DP.get_file_list(
            config)
        if mode == 'train':
            self.data_list = train_list
        # elif mode == 'eval':
        #     self.data_list = val_list
        elif mode == 'test':
            self.data_list = test_list

        self.data_list = [
            fname for fname in self.data_list
            if fname.split('/')[-1] not in invalid_file_list
        ]

        # self.data_list = self.data_list[0]
        # self.data_list = DP.shuffle_list(self.data_list)

        self.label_values = np.sort(
            [k for k, v in config.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        config.ignored_label_inds = [
            self.label_to_idx[ign_label] for ign_label in self.ignored_labels
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        short_name, selected_pc, selected_labels, selected_idx, masks = self.spatially_regular_gen(
            item)
        return short_name, selected_pc, selected_labels, selected_idx, masks

    def spatially_regular_gen(self, item):
        # Generator loop
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
            masks = np.zeros((selected_labels.shape[0], self.cfg.num_classes),
                             dtype=np.float32)
            masks[np.arange(selected_labels.shape[0]), selected_labels[:,
                                                                       0]] = 1
        else:
            # selected_labels = []
            # masks = 0
            selected_labels = labels[selected_idx, :].astype('int64')
            # masks: not for instance but for semantic label
            masks = np.zeros((selected_labels.shape[0], self.cfg.num_classes),
                             dtype=np.float32)
            masks[np.arange(selected_labels.shape[0]), selected_labels[:,
                                                                       0]] = 1

        # FIXME should aug after norm?
        # do data augmentation
        if self.mode != 'test':
            selected_pc = self.data_aug(selected_pc)
        # do normalization
        selected_pc, center, dist = self.normalize(selected_pc)

        return short_name, selected_pc, selected_labels, selected_idx, masks

    @staticmethod
    def get_data(fname):
        short_name = fname.split('/')[-1]
        scan = pcl.load_XYZI(fname)
        xyzi = []
        for i in range(scan.width):
            xyzi.append(scan.__getitem__(i))
        xyzi = np.array(xyzi)
        coords = xyzi[:, :3]
        labels = xyzi[:, 3]
        labels = labels.reshape(-1, 1)

        return short_name, coords, labels

    def collate_fn(self, batch):
        short_name, coords, labels, selected_idx, masks = [], [], [], [], []
        for i in range(len(batch)):
            short_name.append(batch[i][0])
            coords.append(batch[i][1])  # point cloud
            labels.append(batch[i][2])  # index
            selected_idx.append(batch[i][3])  # frame
            masks.append(batch[i][4])  # masks
        coords = np.stack(coords)
        labels = np.stack(labels)
        selected_idx = np.stack(selected_idx)
        masks = np.stack(masks)

        inputs = self.tf_map(coords, labels, selected_idx, masks)

        return short_name, inputs
