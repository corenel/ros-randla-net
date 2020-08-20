import numpy as np

from .base import BaseDataset


class QdhInferenceDataset(BaseDataset):
    def __init__(self, config, mode, pcs, batch_size=5):
        super(QdhInferenceDataset, self).__init__(config, mode)
        self.name = 'qdh_inference'
        self.pcs = pcs
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        selected_pc, selected_idx, = self.spatially_regular_gen(index)
        return selected_pc, selected_idx

    def set_pcs(self, pcs):
        self.pcs = pcs

    def spatially_regular_gen(self, index):
        pick_idx = np.random.choice(len(self.pcs), 1)
        selected_idx = self.crop_pc(self.pcs, pick_idx)
        selected_pc = self.pcs[selected_idx, :]
        return selected_pc, selected_idx

    def collate_fn(self, batch):
        coords, selected_indices = [], []
        for i in range(len(batch)):
            coords.append(batch[i][0])
            selected_indices.append(batch[i][1])
        coords = np.stack(coords)
        selected_indices = np.stack(selected_indices)

        inputs = self.tf_map(batch_pc=coords,
                             batch_label=None,
                             batch_sel_idx=None,
                             batch_masks=None)

        return inputs, selected_indices
