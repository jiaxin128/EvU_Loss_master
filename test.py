#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import warnings
import os
from datetime import datetime
import torch
import logging
from utils.logger import setlogger
from utils.test_utils import test_utils

warnings.filterwarnings("ignore")

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--class_prior', type=list, default=[1.0, 0.005, 0.005, 0.005, 0.005])
    parser.add_argument('--data_dir', type=str, default='/mnt/fdisk/rjx/RESS/data/SEU',
                        help='the directory of the data')
    parser.add_argument('--dataset', type=str, default='SEU', choices=['HYFJ', 'SEU'])
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--model_name', type=str, default='resnet18_1d', help='the name of the model')
    # parser.add_argument('--save_dir', type=str,
    #                     default='/mnt/fdisk/rjx/RESS/checkpoint/SEU/[1.0, 0.1, 0.05, 0.02, 0.005]_AUCEvU/1229-105114/',
    #                     help='the directory to save the model')
    # parser.add_argument('--model_path', type=str, default='33-59.38.pth')

    parser.add_argument('--method', type=str, choices=['EDL', 'AUCEvU'], default='AUCEvU')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/SEU',
                        help='the directory to save the model')
    parser.add_argument('--threshold', type=float, default=0.9)  # 0.1
    parser.add_argument('--post', type=bool, default=True)  # False True

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    dirs = [
        d for d in os.listdir(os.path.join(args.checkpoint_dir))
        if args.method in d and os.path.isdir(os.path.join(args.checkpoint_dir, d))]

    for d in dirs:
        sub_dir = os.path.join(args.checkpoint_dir, d)
        print(sub_dir)

        # folder_name = os.path.basename(sub_dir)
        # prefix = folder_name.rsplit("_", 1)[0]
        #
        # # print(prefix)
        # #
        # # if prefix != "bilstm_1d":
        # #     continue
        #
        # args.model_name = prefix

        for name in sorted(os.listdir(sub_dir)):
            save_dir = os.path.join(sub_dir, name)

            if not os.path.isdir(save_dir):
                continue

            model_path = max(
                (f for f in os.listdir(save_dir) if f.endswith(".pth")),
                key=lambda x: int(x.split("-")[0])
            )

            print(save_dir, '\n')
            args.save_dir = save_dir
            args.model_path = model_path
            if os.path.isdir(save_dir):
                tester = test_utils(args)
                tester.setup()
                if args.post == True:
                    watermark = tester.fit()
                    tester.test(watermark)
                    tester.test_ood(watermark)
                    tester.detection()
                else:
                    tester.test()
                    tester.test_ood()
                    tester.detection()

                # tester.plot_ood_uncertainty_distribution()
                # tester.plot_in_uncertainty_distribution()
                # tester.plot_point_distribution()


"""
    args = parse_args()
    setlogger(os.path.join(args.save_dir, 'test.log'))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    tester = test_utils(args)
    tester.setup()

    if args.post == True:
        watermark = tester.fit()
        tester.test(watermark)
        tester.test_ood(watermark)
    else:
        tester.test()
        tester.test_ood()

    tester.plot_ood_uncertainty_distribution()
    tester.plot_in_uncertainty_distribution()
    tester.plot_point_distribution()
"""
