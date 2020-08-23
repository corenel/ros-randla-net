import argparse
import datetime
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import SubsetRandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from configs import ConfigQDH as cfg
from datasets.qdh import QdhDataset
from utils.loss_utils import compute_acc, IoUCalculator
from utils.network_utils import load_network
from utils.data_utils import make_cuda, visualize
from utils import loss_utils

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
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='trainng device')
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # ===============Create dataset===================
    train_dataset = QdhDataset(cfg, mode='train')
    test_dataset = QdhDataset(cfg, mode='test')
    print('#train: {}'.format(len(train_dataset)))
    print('#eval: {}'.format(len(test_dataset)))

    # num_pcs = []
    # num_labels = defaultdict(int)
    # invalid_files = []
    # for idx in tqdm(range(len(test_dataset))):
    #     short_name, pc, labels, _, _ = test_dataset[idx]
    #     num_pcs.append(pc.shape[0])
    #     if pc.shape[0] < cfg.num_points:
    #         invalid_files.append(short_name)
    #     for label in labels:
    #         num_labels[int(label)] += 1
    # # max=32230 min=5918 avg=14724
    # print('#points: max({}), min({}), avg({})'.format(np.max(num_pcs),
    #                                                   np.min(num_pcs),
    #                                                   np.average(num_pcs)))
    # print('#labels: {}'.format(num_labels))
    # print('#invalid: {}'.format(len(invalid_files)))
    # with open(os.path.join(cfg.root, 'invalid_files.txt'), 'w') as f:
    #     f.writelines(invalid_files)
    # return

    tb_writer = SummaryWriter(logdir=os.path.join(args.logdir))

    # ===================Resume====================
    model, optimizer, start_epoch, scheduler = load_network(
        cfg, args.device, args.checkpoint)
    torch.backends.cudnn.benchmark = True

    # ======================Start==========================
    criterion = nn.CrossEntropyLoss(reduction='none')

    def train_one_epoch(epoch_idx):
        global best_loss
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=cfg.batch_size,
                                       shuffle=True,
                                       collate_fn=train_dataset.collate_fn,
                                       num_workers=cfg.num_workers,
                                       pin_memory=True,
                                       drop_last=False)
        # indices = np.random.choice(range(len(test_dataset)),
        #                            len(test_dataset) // 10)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=cfg.eval_batch_size,
            # sampler=SubsetRandomSampler(indices),
            num_workers=cfg.num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            shuffle=False,
            drop_last=False)
        test_loader_iterator = iter(test_loader)
        start = datetime.datetime.now()
        main_index = 0
        all_loss = 0
        model.train()
        iou_calc = IoUCalculator(cfg)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            train_loader = tqdm(train_loader, ncols=100)
        for batch_idx, (short_name, inputs) in enumerate(train_loader):
            batch_size = len(short_name)
            inputs = make_cuda(inputs)
            f_out = model(inputs)
            # logits = f_out.transpose(1, 2).reshape(-1, cfg.num_classes)
            # labels = inputs['labels'].reshape(-1)
            # loss = criterion(logits, labels).mean()
            loss, logits, labels = loss_utils.compute_loss_simple(
                logits=f_out, labels=inputs['labels'], cfg=cfg)
            iou_calc.add_data(logits, labels)
            acc = compute_acc(logits, labels, cfg)
            main_index += batch_size
            all_loss += loss * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb_writer.add_scalar(
                'acc', float(acc),
                int(epoch_idx) * len(train_loader) * int(cfg.batch_size) +
                main_index)
            tb_writer.add_scalar(
                'logits', float(loss),
                int(epoch_idx) * len(train_loader) * int(cfg.batch_size) +
                main_index)
            tb_writer.add_scalar(
                'sum_loss', all_loss / main_index,
                int(epoch_idx) * len(train_loader) * int(cfg.batch_size) +
                main_index)

            if epoch_idx >= cfg.max_epoch // 10 and \
                    batch_idx % 8 == 0 and \
                    cfg.use_full_set_pc_in_training:
                del short_name
                del inputs
                optimizer.zero_grad()
                for test_batch_idx in range(8):
                    try:
                        test_batch = next(test_loader_iterator)
                    except StopIteration:
                        test_loader_iterator = iter(test_loader)
                        test_batch = next(test_loader_iterator)
                    short_name, inputs = test_batch
                    inputs = make_cuda(inputs)
                    f_out = model(inputs)
                    test_loss, logits, labels = loss_utils.compute_loss_simple(
                        logits=f_out, labels=inputs['labels'], cfg=cfg)
                    # logits = f_out.transpose(1, 2).reshape(-1, cfg.num_classes)
                    # labels = inputs['labels'].reshape(-1)
                    # test_loss = criterion(logits, labels).mean()
                    test_acc = compute_acc(logits, labels, cfg)
                    test_loss.backward()
                    epochs.set_description(
                        "Epoch (Loss=%g,TestLoss=%g,TestAcc=%g)" % (
                            round(loss.item(), 5),
                            round(test_loss.item(), 5),
                            round(test_acc.item(), 5),
                        ))
                optimizer.step()
            else:
                epochs.set_description(
                    "Epoch (Loss=%g,Acc=%g)" %
                    (round(loss.item(), 5), round(acc.item(), 5)))

        now = datetime.datetime.now()
        duration = now - start
        log = '> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |'
        log = log.format(now.strftime("%c"), epoch_idx, cfg.max_epoch,
                         duration.total_seconds())
        log += 'loss: {:.4f} |'.format(all_loss / len(train_dataset))

        if all_loss / len(train_dataset) < best_loss:
            best_loss = all_loss / len(train_dataset)
        fname = os.path.join(args.logdir, 'model' + str(epoch_idx) + '.pth')
        print('> Saving model to {}...'.format(fname))
        torch.save(
            {
                'epoch': epoch_idx + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, fname)

        log += ' best: {:.4f} |'.format(best_loss)
        mean_iou, iou_list = iou_calc.compute_iou()
        log += 'mean IoU:{:.1f} |'.format(mean_iou * 100)
        s = 'IoU: '
        for iou_tmp in iou_list:
            s += '{:5.2f} |'.format(100 * iou_tmp)
        log += s

        fname = os.path.join(args.logdir, 'train.log')
        with open(fname, 'a') as fp:
            fp.write(log + '\n')

        print(log)

    def eval_one_epoch(epoch_idx):
        model.eval()
        main_index = 0
        with torch.no_grad():
            test_loader = data.DataLoader(test_dataset,
                                          batch_size=cfg.eval_batch_size,
                                          num_workers=cfg.num_workers,
                                          collate_fn=test_dataset.collate_fn,
                                          pin_memory=True,
                                          shuffle=False,
                                          drop_last=False)
            # eval_scalars = defaultdict(list)
            eval_iou_calc = IoUCalculator(cfg)
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
                test_loader = tqdm(test_loader, ncols=100)
            for i, (short_name, inputs) in enumerate(test_loader):
                batch_size = len(short_name)
                inputs = make_cuda(inputs)
                f_out = model(inputs)
                logits = f_out.transpose(1, 2).reshape(-1, cfg.num_classes)
                labels = inputs['labels'].reshape(-1)
                # _, valid_logits, valid_labels = loss_utils.compute_loss_simple(
                #     logits=f_out, labels=inputs['labels'], cfg=cfg)
                eval_iou_calc.add_data(logits, labels)
                acc = compute_acc(logits, labels, cfg)
                # loss = loss1 + loss2
                main_index += batch_size
                epochs.set_description("Epoch (Acc=%g)" %
                                       (round(acc.item(), 5)))

            eval_log = '> Epoch [{:04d}/{:04d}] | Eval |'.format(
                epoch_idx, cfg.max_epoch)
            eval_mean_iou, eval_iou_list = eval_iou_calc.compute_iou()
            eval_log += 'mean IoU:{:.1f} |'.format(eval_mean_iou * 100)
            s = 'IoU: '
            for iou_tmp in eval_iou_list:
                s += '{:5.2f} |'.format(100 * iou_tmp)
            eval_log += s

            tb_writer.add_scalar(
                'eval_miou', eval_mean_iou,
                int(epoch_idx) * len(test_loader) * int(cfg.batch_size))

            fname = os.path.join(args.logdir, 'eval.log')
            with open(fname, 'a') as fp:
                fp.write(eval_log + '\n')

            print(eval_log)

    def visualize_one_epoch(i_epoch):
        model.eval()
        with torch.no_grad():
            test_loader = data.DataLoader(test_dataset,
                                          batch_size=cfg.eval_batch_size,
                                          num_workers=cfg.num_workers,
                                          collate_fn=test_dataset.collate_fn,
                                          pin_memory=True,
                                          shuffle=False,
                                          drop_last=False)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            test_loader = tqdm(test_loader, ncols=100)
        for i, (short_name, inputs) in enumerate(test_loader):
            if i > 10:
                break
            inputs = make_cuda(inputs)
            f_out = model(inputs)
            logits = f_out.transpose(1, 2).reshape(-1, cfg.num_classes)
            pred = logits.max(dim=1)[1]
            visualize(inputs['xyz'][0], pred,
                      '{:03d}_{:02d}.pcd'.format(i_epoch, i))

    # with torchsnooper.snoop():
    epochs = trange(cfg.max_epoch, leave=True, desc="Epoch")
    for i_epoch in range(start_epoch, cfg.max_epoch):
        train_one_epoch(i_epoch)
        if i_epoch % 5 == 0:
            eval_one_epoch(i_epoch)
            # visualize_one_epoch(i_epoch)
        scheduler.step()
        epochs.update()
        print('-' * 30)


if __name__ == "__main__":
    embed()
