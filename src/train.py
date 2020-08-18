import argparse
import datetime
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from configs import ConfigQDH as cfg
from datasets.qdh import QdhDataset
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
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='trainng device')
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    # ===============Create dataset===================
    train_dataset = QdhDataset(cfg, mode='train')
    tb_writer = SummaryWriter(logdir=os.path.join(args.logdir))

    # ===================Resume====================
    model, optimizer, start_epoch, scheduler = load_network(
        cfg, args.device, args.checkpoint)

    # ======================Start==========================
    criterion = nn.CrossEntropyLoss(reduction='none')

    def train_one_epoch(epoch_idx):
        global best_loss
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=cfg.batch_size,
                                       shuffle=True,
                                       collate_fn=train_dataset.collate_fn,
                                       num_workers=cfg.num_workers,
                                       drop_last=False)
        start = datetime.datetime.now()
        scheduler.step()
        main_index = 0
        all_loss = 0
        model.train()
        iou_calc = IoUCalculator(cfg)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            train_loader = tqdm(train_loader, ncols=100)
        for i, (short_name, inputs) in enumerate(train_loader):
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
            loss = criterion(logits, labels)
            loss = loss.mean()
            iou_calc.add_data(logits, labels)
            acc = compute_acc(logits, labels)
            main_index += batch_size
            all_loss += loss * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb_writer.add_scalar(
                'acc', acc,
                int(epoch_idx) * len(train_loader) * int(cfg.batch_size) +
                main_index)
            tb_writer.add_scalar(
                'logits', loss,
                int(epoch_idx) * len(train_loader) * int(cfg.batch_size) +
                main_index)
            tb_writer.add_scalar(
                'sum_loss', all_loss / main_index,
                            int(epoch_idx) * len(train_loader) * int(cfg.batch_size) +
                            main_index)
            epochs.set_description("Epoch (Loss=%g)" % round(loss.item(), 5))

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

    # def eval_one_epoch(i_epoch):
    #     model.eval()
    #     eval_loss = 0
    #     main_index = 0
    #     with torch.no_grad():
    #         test_loader = data.DataLoader(test_dataset,
    #                                       batch_size=cfg.batch_size,
    #                                       num_workers=cfg.num_workers,
    #                                       collate_fn=train_dataset.collate_fn,
    #                                       shuffle=False,
    #                                       drop_last=False)
    #         eval_scalars = defaultdict(list)
    #         eval_iou_calc = IoUCalculator(cfg)
    #         if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
    #             test_loader = tqdm(test_loader, ncols=100)
    #         for i, (short_name, inputs) in enumerate(test_loader):
    #             batch_size = len(short_name)
    #             for key in inputs:
    #                 if type(inputs[key]) is list:
    #                     for i in range(len(inputs[key])):
    #                         inputs[key][i] = inputs[key][i].cuda()
    #                 else:
    #                     inputs[key] = inputs[key].cuda()
    #
    #             f_out = model(inputs)
    #             # loss1 = criterion['discriminative'](embedding, inputs["masks"], inputs["labels"])
    #             loss, valid_logits, valid_labels = Logits(
    #                 f_out, inputs['labels'], cfg)
    #
    #             acc = compute_acc(valid_logits, valid_labels)
    #             eval_iou_calc.add_data(valid_logits, valid_labels)
    #             # loss = loss1 + loss2
    #             main_index += batch_size
    #             eval_loss += loss * batch_size
    #
    #         eval_log = '> | Eval |'
    #         eval_log += 'eval_loss : {:.4f} |'.format(eval_loss /
    #                                                   len(test_dataset))
    #         eval_mean_iou, eval_iou_list = eval_iou_calc.compute_iou()
    #         eval_log += 'mean IoU:{:.1f} |'.format(eval_mean_iou * 100)
    #         s = 'IoU: '
    #         for iou_tmp in eval_iou_list:
    #             s += '{:5.2f} |'.format(100 * iou_tmp)
    #         eval_log += s
    #
    #         tb_writer.add_scalar(
    #             'eval_loss', (eval_loss) / len(test_dataset),
    #                          int(i_epoch) * len(test_loader) * int(cfg.batch_size))
    #         tb_writer.add_scalar(
    #             'eval_miou', eval_mean_iou,
    #             int(i_epoch) * len(test_loader) * int(cfg.batch_size))
    #
    #         fname = os.path.join(args.logdir, 'train.log')
    #         with open(fname, 'a') as fp:
    #             fp.write(eval_log + '\n')
    #
    #         print(eval_log)

    # with torchsnooper.snoop():
    epochs = trange(cfg.max_epoch, leave=True, desc="Epoch")
    for i_epoch in range(start_epoch, cfg.max_epoch):
        train_one_epoch(i_epoch)
        print('-' * 30)


if __name__ == "__main__":
    embed()
