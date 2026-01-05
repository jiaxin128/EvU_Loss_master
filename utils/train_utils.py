import logging
import os
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import optim
from tqdm import tqdm

import models
from loss.loss import edl_digamma_loss, exp_evidence, AUCEvULoss
from utils.tools import one_hot_embedding

import torch.nn.functional as F


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.best_acc = 0

    def setup(self):
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        self.datasets = {}
        from datasets.SEU import Mechanical_datasets
        self.datasets['train'] = Mechanical_datasets(args.data_dir, args.class_prior, args.normlizetype).data_preprare(
            'train')
        self.datasets['val'] = Mechanical_datasets(args.data_dir, args.class_prior, args.normlizetype).data_preprare(
            'val')
        self.datasets['test'] = Mechanical_datasets(args.data_dir, args.class_prior, args.normlizetype).data_preprare(
            'test')
        self.datasets['ood'] = Mechanical_datasets(args.data_dir, args.class_prior, args.normlizetype).data_preprare(
            'ood')

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=True)
                            for x in ['train', 'val', 'test', 'ood']}

        self.inputchannel = Mechanical_datasets.inputchannel
        self.num_classes = Mechanical_datasets.num_classes
        self.model = getattr(models, args.model_name)(in_channel=self.inputchannel, out_channel=self.num_classes)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        if args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                       weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0

        self.model.to(self.device)
        if args.method == 'EDL':
            self.criterion = edl_digamma_loss
        else:
            self.criterion = edl_digamma_loss
            self.evu_criterion = AUCEvULoss()

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            if self.lr_scheduler is not None:
                logging.info('current G_lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current G_lr: {}'.format(args.lr))
            self.model.train()
            for batch_idx, (inputs, labels) in enumerate(self.dataloaders['train']):
                inputs = inputs.to(self.device)
                one_labels = one_hot_embedding(labels, self.num_classes)
                one_labels = one_labels.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, pred = torch.max(outputs, 1)
                if args.method == 'EDL':
                    loss = self.criterion(outputs, one_labels.float(), epoch, self.num_classes, 100, self.device)
                else:
                    evu_loss = self.evu_criterion(outputs, labels, self.num_classes)
                    edl_loss = self.criterion(outputs, one_labels.float(), epoch, self.num_classes, 100, self.device)
                    loss = edl_loss + args.delta * evu_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.validate(epoch)
            self.model.train()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self, dataloader, watermark):
        self.model.eval()
        all_alpha, all_prob, all_labels, all_un = [], [], [], []

        for inputs, labels in dataloader:
            if watermark is not None:
                watermark = watermark.to(self.device, non_blocking=True)
                inputs = inputs.to(self.device, non_blocking=True)
                inputs = inputs + watermark
            else:
                inputs = inputs.to(self.device, non_blocking=True)
            outputs = self.model(inputs)
            evidence = exp_evidence(outputs)
            alpha = evidence + 1
            prob = torch.max(alpha, dim=1)[0] / alpha.sum(dim=1)
            u = self.num_classes / alpha.sum(dim=1, keepdim=True)

            all_alpha.append(alpha.cpu())
            all_prob.append(prob.cpu())
            all_labels.append(labels)
            all_un.append(u.cpu())

        all_pred = torch.cat(all_alpha, dim=0)
        all_prob = torch.cat(all_prob, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_un = torch.cat(all_un, dim=0)

        min_un = torch.min(all_un)
        max_un = torch.max(all_un)

        all_un = (all_un - min_un) / (max_un - min_un + 1e-8)

        pred_labels = torch.argmax(all_pred, dim=1)

        pred_np = pred_labels.numpy()
        prob_np = all_prob.numpy()
        y_np = all_labels.numpy()
        un_np = all_un.numpy().squeeze()

        return (
            pred_np,
            prob_np,
            y_np,
            un_np
        )

    def validate(self, epoch):
        self.model.eval()
        test_pred, test_prob, test_labels, test_un = self.evaluate(self.dataloaders['test'], watermark=None)
        accuracy_test = accuracy_score(test_labels, test_pred)
        logging.info('Test: -Accuracy: {:.2%}'.format(accuracy_test))

        model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
        if accuracy_test > self.best_acc:
            self.best_acc = accuracy_test
            save_msg = 'Save epoch {} -Accuracy: {:.2%}'.format(epoch, accuracy_test)
            self.save_name_best = '{}-{:.2f}.pth'.format(epoch, accuracy_test * 100)
            torch.save(model_state_dic, os.path.join(self.save_dir, self.save_name_best))
            logging.info(save_msg)

    def load_model(self):
        model_name = max(
            (f for f in os.listdir(self.save_dir) if f.endswith(".pth")),
            key=lambda x: int(x.split("-")[0])
        )
        model_path = os.path.join(self.save_dir, model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def fit(self):
        args = self.args
        self.load_model()
        watermark = torch.zeros(1, 1, 1024).cuda()
        watermark.requires_grad_()
        for _ in tqdm(range(50)):
            for inputs, labels in self.dataloaders['val']:
                inputs = inputs.to(self.device)
                one_labels = one_hot_embedding(labels, self.num_classes)
                one_labels = one_labels.to(self.device)

                inputs_per = torch.randn_like(inputs) * 1
                inputs_per = torch.flip(inputs_per, dims=[-1])
                inputs_per = torch.roll(inputs_per, shifts=8, dims=-1)

                inputs_per = torch.clamp(inputs_per, 0, 1)
                data = torch.cat([inputs, inputs_per], dim=0)

                watermark.requires_grad_()
                self.model.zero_grad()
                outputs = self.model(data + watermark)
                evidence = exp_evidence(outputs) / 1.0  # 2.0
                alpha = evidence + 1
                uncertain = self.num_classes / alpha.sum(dim=1, keepdim=True)
                loss = self.criterion(outputs[:len(inputs)], one_labels.float(), 0, self.num_classes, 1, self.device)
                # loss += torch.relu(0.7 - uncertain[:len(inputs)].mean() + uncertain[len(inputs):].mean())
                loss += torch.relu(0.7 + uncertain[len(inputs):].mean())

                grad = torch.autograd.grad(loss, [watermark])[0].detach()
                perturb = grad.sign() * grad.abs() / (grad.pow(2).sum().sqrt() + 1e-8)

                watermark.requires_grad_()
                self.model.zero_grad()
                outputs = self.model(data + watermark + perturb * 0.02)
                evidence = exp_evidence(outputs) / 1.0  # 2.0
                alpha = evidence + 1
                uncertain = self.num_classes / alpha.sum(dim=1, keepdim=True)
                loss_ = self.criterion(outputs[:len(inputs)], one_labels.float(), 0, self.num_classes, 1, self.device)
                loss_ += torch.relu(0.2 - uncertain[:len(inputs)].mean() + uncertain[len(inputs):].mean())
                grad = torch.autograd.grad(loss_, [watermark])[0].detach()
                watermark = watermark.detach() - torch.sign(grad) * 0.0005
        return watermark

    def test(self, watermark=None):
        self.load_model()
        pred_np, prob_np, y_np, un_np = self.evaluate(self.dataloaders['test'], watermark)
        accuracy = accuracy_score(y_np, pred_np)
        logging.info('Final Result')
        logging.info('Test: -Accuracy: {:.2%}'.format(accuracy))

        if watermark is not None:
            test_np_path = os.path.join(self.save_dir, "test_post.npz")
        else:
            test_np_path = os.path.join(self.save_dir, "test.npz")
        np.savez(test_np_path, pred=pred_np, prob=prob_np, label=y_np, uncertainty=un_np)

    def test_ood(self, watermark=None):
        args = self.args
        self.load_model()
        pred_np, prob_np, y_np, un_np = self.evaluate(self.dataloaders['ood'], watermark)
        if watermark is not None:
            ood_np_path = os.path.join(self.save_dir, "ood_post.npz")
        else:
            ood_np_path = os.path.join(self.save_dir, "ood.npz")

        np.savez(ood_np_path, pred=pred_np, prob=prob_np, label=y_np, uncertainty=un_np)

    def detection(self, watermark=None):
        if watermark is not None:
            test_path = os.path.join(self.save_dir, "test_post.npz")
            ood_path = os.path.join(self.save_dir, "ood_post.npz")
        else:
            test_path = os.path.join(self.save_dir, "test.npz")
            ood_path = os.path.join(self.save_dir, "ood.npz")

        test_data = np.load(test_path)
        ood_data = np.load(ood_path)

        uncertainty_id = test_data["uncertainty"].flatten()
        uncertainty_ood = ood_data["uncertainty"].flatten()

        all_uncertainty = np.concatenate([uncertainty_id, uncertainty_ood])
        min_u, max_u = all_uncertainty.min(), all_uncertainty.max()

        uncertainty_list = []

        for u in uncertainty_id:
            norm_u = (u - min_u) / (max_u - min_u)
            uncertainty_list.append({'type': 'ID', 'uncertainty': norm_u})

        for u in uncertainty_ood:
            norm_u = (u - min_u) / (max_u - min_u)
            uncertainty_list.append({'type': 'OOD', 'uncertainty': norm_u})

        scores = np.concatenate([uncertainty_id, uncertainty_ood])
        labels = np.concatenate([np.zeros_like(uncertainty_id), np.ones_like(uncertainty_ood)])

        threshold = np.percentile(uncertainty_id, 95)
        preds = (scores > threshold).astype(int)
        acc = accuracy_score(labels, preds)

        logging.info('OOD: -Detection Accuracy: {:.2%}'.format(acc))
