#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils import train_utils
import warnings
import time

warnings.filterwarnings("ignore")

args = None


# 1 0.1 0.05 0.02 0.005
# 1 0.25 0.165 0.08 0.03
# 1 0.5 0.33 0.16 0.083


# 1 0.1 0.1 0.1 0.1
# 1 0.02 0.02 0.02 0.02
# 1 0.01 0.01 0.01 0.01
# 1 0.005 0.005 0.005 0.005


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='resnet18_1d', help='the name of the model')
    parser.add_argument('--class_prior', type=float, nargs='+', default=[1, 0.005, 0.005, 0.005, 0.005])
    parser.add_argument('--dataset', type=str, default='MGB', choices=['HYFJ', 'SEU', 'MGB'])
    # parser.add_argument('--data_dir', type=str, default='/mnt/fdisk/rjx/RESS/data/MGB',
    #                     help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--cuda_device', type=str, default='2', help='assign device')
    # parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/MGB1',
    #                     help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='7', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')
    parser.add_argument('--method', type=str, choices=['EDL', 'AUCEvU'], default='AUCEvU')
    parser.add_argument('--threshold', type=float, default=0.9)  # 0.1
    parser.add_argument('--delta', type=float, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    args.data_dir = os.path.join('/mnt/fdisk/rjx/RESS/data/', args.dataset)
    args.checkpoint_dir = os.path.join('/mnt/fdisk/rjx/RESS/checkpoint/', args.dataset + '1')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    sub_dir = str(args.class_prior) + '_' + args.method
    time_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir, time_dir)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    t0 = time.time()
    trainer.setup()
    trainer.train()
    t1 = time.time()
    print(f"[Time] setup: {t1 - t0:.2f}s")
    trainer.test()
    trainer.test_ood()
    trainer.detection()
    t2 = time.time()
    print(f"[Time] setup: {t2 - t1:.2f}s")
    watermark = trainer.fit()
    t3 = time.time()
    print(f"[Time] setup: {t3 - t2:.2f}s")
    trainer.test(watermark)
    trainer.test_ood(watermark)
    trainer.detection(watermark)
    t4 = time.time()
    print(f"[Time] setup: {t4 - t3:.2f}s")





    """
    # sub_dir = str(args.model_name) + '_' + args.method
    # sub_dir = str(args.class_prior) + '_' + 'Softplus'
    time_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir, time_dir)
    os.makedirs(save_dir, exist_ok=True)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
"""