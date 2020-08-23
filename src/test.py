import argparse
import os

import numpy as np
import pcl
import torch
import torch.utils.data as data

from configs import ConfigQDH as cfg
from datasets.qdh_inference import QdhInferenceDataset
from utils.data_utils import make_cuda, visualize, XYZRGB_to_XYZ
from utils.network_utils import load_network


def embed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        type=str,
                        required=True,
                        help='path to checkpoint file')
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help='path to input PCD file')
    parser.add_argument('--output', type=str, help='path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='trainng device')
    args = parser.parse_args()

    # ===============Create dataset===================

    pcs = pcl.load_XYZI(args.input)
    pcs_xyz = XYZRGB_to_XYZ(pcs)
    pcs_np = np.asarray(pcs_xyz)
    test_dataset = QdhInferenceDataset(cfg, mode='inference', pcs=pcs_np)

    # ===================Resume====================
    model, optimizer, start_epoch, scheduler = load_network(
        cfg, args.device, args.checkpoint)
    model.eval()

    # ======================Start==========================
    with torch.no_grad():
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=1,
                                      num_workers=1,
                                      collate_fn=test_dataset.collate_fn,
                                      shuffle=False,
                                      drop_last=False)
        batch_data = next(iter(test_loader))
        inputs, selected_indices = batch_data
        inputs = make_cuda(inputs)
        preds = model(inputs)
        logits = preds
        logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
        preds = logits.max(dim=1)[1].cpu().numpy()
        if args.output is not None:
            outpath = args.output
        else:
            outpath = os.path.join('results', os.path.basename(args.input))
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
        visualize(preds, pcs_xyz, selected_indices, outpath)
        print('result is saved to {}'.format(outpath))


if __name__ == "__main__":
    embed()
