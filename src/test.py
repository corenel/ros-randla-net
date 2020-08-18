import argparse
import logging

import numpy as np
import open3d
import torch.utils.data as data
from tqdm import tqdm

from configs import ConfigQDH as cfg
from datasets.qdh import qdhset
from models.RandLANet import *
from utils.loss_utils import compute_acc, IoUCalculator
from utils.network_utils import load_network

best_loss = np.Inf


def embed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',
                        type=str,
                        default='logs/qdh_roi',
                        help='path to the logging directory')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='path to checkpoint file')
    args = parser.parse_args()

    # ===============Create dataset===================
    test_dataset = qdhset(cfg, mode='test')

    # ===================Resume====================
    model, optimizer, start_epoch, scheduler = load_network(
        cfg, args.device, args.checkpoint)
    model.eval()

    # ======================Start==========================
    criterion = nn.CrossEntropyLoss(reduction='none')

    eval_loss = 0
    main_index = 0
    with torch.no_grad():
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=1,
                                      num_workers=cfg.num_workers,
                                      collate_fn=test_dataset.collate_fn,
                                      shuffle=False,
                                      drop_last=False)
        eval_iou_calc = IoUCalculator(cfg)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            test_loader = tqdm(test_loader, ncols=100)
        for i, (short_name, inputs) in enumerate(test_loader):
            batch_size = len(short_name)
            for key in inputs:
                if type(inputs[key]) is list:
                    for i in range(len(inputs[key])):
                        inputs[key][i] = inputs[key][i].cuda()
                else:
                    inputs[key] = inputs[key].cuda()

            f_out = model(inputs)
            logits = f_out.transpose(1, 2).reshape(-1, cfg.num_classes)
            labels = inputs['labels'].reshape(-1)
            acc = compute_acc(logits, labels)
            pred = logits.max(dim=1)[1]
            print(short_name, 'Acc: ', acc)

            visualize(inputs['xyz'][0], pred, short_name)


def visualize(xyz, pred, short_name):
    color = [[245 / 255, 150 / 255, 100 / 255],
             [90 / 255, 30 / 255, 150 / 255], [80 / 255, 240 / 255, 150 / 255]]
    xyz = xyz.squeeze().cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    pred_label_set = list(set(pred))
    pred_label_set.sort()
    print(pred_label_set)
    viz_point = open3d.PointCloud()
    point_cloud = open3d.PointCloud()
    for id_i, label_i in enumerate(pred_label_set):
        # print('sem_label:', label_i, )
        index = np.argwhere(pred == label_i).reshape(-1)
        sem_cluster = xyz[index, :]
        point_cloud.points = open3d.Vector3dVector(sem_cluster)
        point_cloud.paint_uniform_color(color[id_i])
        viz_point += point_cloud

    open3d.draw_geometries([viz_point],
                           window_name=short_name[0],
                           width=1920,
                           height=1080,
                           left=50,
                           top=50)


if __name__ == "__main__":
    embed()
