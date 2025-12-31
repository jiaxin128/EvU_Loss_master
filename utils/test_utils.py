import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import models
from datasets import SEU
from loss.loss import edl_digamma_loss, exp_evidence
from utils.metric import evidence_vs_uncertainty, compute_calibration
from utils.tools import one_hot_embedding


class test_utils(object):
    def __init__(self, args):
        self.args = args

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
        self.datasets['ood'] = Mechanical_datasets('/mnt/fdisk/rjx/RESS/data/SEU', args.class_prior,
                                                   args.normlizetype).data_preprare('ood')

        if args.dataset == 'HYFJ':
            from datasets.HYFJ import Mechanical_datasets
        elif args.dataset == 'MGB':
            from datasets.MGB import Mechanical_datasets
        else:
            from datasets.SEU import Mechanical_datasets

        # self.datasets = {}
        self.datasets['train'] = Mechanical_datasets(args.data_dir, args.class_prior, args.normlizetype).data_preprare(
            'train')
        self.datasets['val'] = Mechanical_datasets(args.data_dir, args.class_prior, args.normlizetype).data_preprare(
            'val')
        self.datasets['test'] = Mechanical_datasets(args.data_dir, args.class_prior, args.normlizetype).data_preprare(
            'test')
        # self.datasets = {}
        #
        # if args.dataset == 'HYFJ':
        #     from datasets.HYFJ import Mechanical_datasets
        #     self.datasets['val'] = Mechanical_datasets(args.data_dir, args.class_prior,
        #                                                  args.normlizetype).data_preprare('val')
        #     self.datasets['test'] = Mechanical_datasets(args.data_dir, args.class_prior,
        #                                                 args.normlizetype).data_preprare('test')
        #     self.datasets['ood'] = SEU.Mechanical_datasets('/mnt/fdisk/rjx/RESS/data/SQI', args.class_prior,
        #                                                    args.normlizetype).data_preprare('ood')
        # else:
        #     from datasets.SEU import Mechanical_datasets
        #     self.datasets['val'] = Mechanical_datasets(args.data_dir, args.class_prior,
        #                                                  args.normlizetype).data_preprare('val')
        #     self.datasets['test'] = Mechanical_datasets(args.data_dir, args.class_prior,
        #                                                 args.normlizetype).data_preprare('test')
        #     self.datasets['ood'] = Mechanical_datasets(args.data_dir, args.class_prior,
        #                                                args.normlizetype).data_preprare('ood')

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['val', 'test', 'ood']}
        self.inputchannel = Mechanical_datasets.inputchannel
        self.num_classes = Mechanical_datasets.num_classes
        self.model = getattr(models, args.model_name)(in_channel=self.inputchannel, out_channel=self.num_classes)
        self.model.to(self.device)
        self.num_classes = Mechanical_datasets.num_classes

        self.criterion = edl_digamma_loss

    def load_model(self):
        args = self.args
        model_path = os.path.join(args.save_dir, args.model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader, watermark=None):
        all_alpha, all_prob, all_labels, all_un = [], [], [], []

        for inputs, labels in dataloader:
            if watermark != None:
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

        alpha = torch.cat(all_alpha)
        prob = torch.cat(all_prob)
        labels = torch.cat(all_labels)
        un = torch.cat(all_un).squeeze()

        pred = torch.argmax(alpha, dim=1)

        return (
            pred.numpy(),
            prob.numpy(),
            labels.numpy(),
            un.numpy()
        )

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
                inputs_per = torch.clamp(inputs_per, 0.0, 1.0)
                data = torch.cat([inputs, inputs_per], dim=0)

                watermark.requires_grad_()
                self.model.zero_grad()
                outputs = self.model(data + watermark)
                evidence = exp_evidence(outputs) / 2.0
                alpha = evidence + 1
                uncertain = self.num_classes / alpha.sum(dim=1, keepdim=True)
                loss = self.criterion(outputs[:len(inputs)], one_labels.float(), 0, self.num_classes, 1, self.device)
                loss += torch.relu(0.2 - uncertain[:len(inputs)].mean() + uncertain[len(inputs):].mean())

                grad = torch.autograd.grad(loss, [watermark])[0].detach()
                perturb = grad.sign() * grad.abs() / (grad.pow(2).sum().sqrt() + 1e-8)

                watermark.requires_grad_()
                self.model.zero_grad()
                outputs = self.model(data + watermark + perturb * 0.1)
                evidence = exp_evidence(outputs) / 2.0
                alpha = evidence + 1
                uncertain = self.num_classes / alpha.sum(dim=1, keepdim=True)
                loss_ = self.criterion(outputs[:len(inputs)], one_labels.float(), 0, self.num_classes, 1, self.device)
                loss_ += torch.relu(0.2 - uncertain[:len(inputs)].mean() + uncertain[len(inputs):].mean())
                grad = torch.autograd.grad(loss_, [watermark])[0].detach()
                watermark = watermark.detach() - torch.sign(grad) * 0.001
        return watermark

    def test(self, watermark=None):
        args = self.args
        self.load_model()
        pred_np, prob_np, y_np, un_np = self.evaluate(self.dataloaders['test'], watermark)
        accuracy = accuracy_score(y_np, pred_np)
        recall = recall_score(y_np, pred_np, average='macro')
        precision = precision_score(y_np, pred_np, average='macro')
        f1 = f1_score(y_np, pred_np, average='macro')

        pac, pui, evu = evidence_vs_uncertainty(pred_np, y_np, un_np, args.threshold)
        avg_acc, ece, uce = compute_calibration(pred_np, prob_np, y_np, un_np)

        if args.post == True:
            ood_result_txt = os.path.join(args.save_dir, 'post_test_result.txt')
            with open(ood_result_txt, 'w') as f:  # 注意是 'w' 覆盖
                f.write(
                    f"Accuracy: {accuracy:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"F1: {f1:.4f}\n"
                    f"PAC: {pac:.4f}\n"
                    f"PUI: {pui:.4f}\n"
                    f"EVU: {evu:.4f}\n"
                    f"UCE: {uce:.4f}\n"
                )

        if args.post == True:
            test_np_path = os.path.join(args.save_dir, "test_post.npz")
        else:
            test_np_path = os.path.join(args.save_dir, "test.npz")
        np.savez(test_np_path, pred=pred_np, prob=prob_np, label=y_np, uncertainty=un_np)

    def test_ood(self, watermark=None):
        args = self.args
        self.load_model()
        pred_np, prob_np, y_np, un_np = self.evaluate(self.dataloaders['ood'], watermark)

        pac, pui, evu = evidence_vs_uncertainty(pred_np, y_np, un_np, args.threshold)
        logging.info('OOD: -PAC: {:.2%}, -PUI: {:.2%}, -EVU: {:.2%}'.format(pac, pui, evu))
        if args.post == True:
            ood_np_path = os.path.join(args.save_dir, "ood_post.npz")
        else:
            ood_np_path = os.path.join(args.save_dir, "ood.npz")

        np.savez(ood_np_path, pred=pred_np, prob=prob_np, label=y_np, uncertainty=un_np)

    def detection(self):
        args = self.args

        if args.post == True:
            test_path = os.path.join(args.save_dir, "test_post.npz")
            ood_path = os.path.join(args.save_dir, "ood_post.npz")
        else:
            test_path = os.path.join(args.save_dir, "test.npz")
            ood_path = os.path.join(args.save_dir, "ood.npz")


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

        df_un = pd.DataFrame(uncertainty_list)
        scores = np.concatenate([uncertainty_id, uncertainty_ood])
        labels = np.concatenate([np.zeros_like(uncertainty_id), np.ones_like(uncertainty_ood)])
        auc = roc_auc_score(labels, scores)

        # print("ID 平均不确定性:", np.mean(uncertainty_id))
        # print("OOD 平均不确定性:", np.mean(uncertainty_ood))

        threshold = np.percentile(uncertainty_id, 95)
        preds = (scores > threshold).astype(int)
        acc = accuracy_score(labels, preds)
        # print("Detection Accuracy (fixed ID threshold):", acc)

        logging.info(
            'OOD: -ROC-AUC: {:.2}, -Detection Accuracy: {:.2%}'.format(auc, acc))
        if args.post == True:
            ood_result_txt = os.path.join(args.save_dir, 'ood_post_test_result.txt')
        else:
            ood_result_txt = os.path.join(args.save_dir, 'ood_test_result.txt')

        with open(ood_result_txt, 'w') as f:  # 注意是 'w' 覆盖
            f.write(
                f"ROC-AUC: {auc:.4f}\n"
                f"Detection Accuracy: {acc:.4f}\n"
            )


    def plot_ood_uncertainty_distribution(self):
        args = self.args
        if args.post == True:
            test_path = os.path.join(args.save_dir, "test_post.npz")
            ood_path = os.path.join(args.save_dir, "ood_post.npz")
        else:
            test_path = os.path.join(args.save_dir, "test.npz")
            ood_path = os.path.join(args.save_dir, "ood.npz")

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

        df_un = pd.DataFrame(uncertainty_list)
        plt.figure(figsize=(6, 6))
        g = sns.displot(
            df_un,
            x="uncertainty",
            hue="type",
            kind="kde",
            fill=True,
            height=4,
        )

        # 获取 legend 对象并调整
        legend = g._legend
        legend.set_bbox_to_anchor((0.8, 0.85))

        # 字体稍大
        legend.set_title("Type")
        legend.get_title().set_fontsize(12)
        for text in legend.texts:
            text.set_fontsize(12)

        for ax in g.axes.flatten():
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        g.set_axis_labels("Normalized Uncertainty", "Density")
        plt.subplots_adjust(right=0.95)

        if args.post == True:
            savename = os.path.join(args.save_dir, 'plot_post_ood_uncertainty_distribution.png')
        else:
            savename = os.path.join(args.save_dir, 'plot_ood_uncertainty_distribution.png')

        # plt.savefig(savename, format='png')
        plt.savefig(savename, dpi=300, bbox_inches="tight", format='png')
        # plt.show()
        plt.close()

    def plot_point_distribution(self):
        args = self.args
        if args.post == True:
            test_path = os.path.join(args.save_dir, "test_post.npz")
            ood_path = os.path.join(args.save_dir, "ood_post.npz")
        else:
            test_path = os.path.join(args.save_dir, "test.npz")
            ood_path = os.path.join(args.save_dir, "ood.npz")

        test_data = np.load(test_path)
        ood_data = np.load(ood_path)
        print(1)
        uncertainty_id = test_data["uncertainty"].flatten()
        labels_id = test_data["label"].flatten()
        uncertainty_ood = ood_data["uncertainty"].flatten()

        all_uncertainty = np.concatenate([uncertainty_id, uncertainty_ood])
        min_u, max_u = all_uncertainty.min(), all_uncertainty.max()

        data_list = []
        for u, l in zip(uncertainty_id, labels_id):
            norm_u = (u - min_u) / (max_u - min_u)
            data_list.append({"type": "id", "uncertainty": norm_u, "label": int(l)})
        for u in uncertainty_ood:
            norm_u = (u - min_u) / (max_u - min_u)
            data_list.append({"type": "ood", "uncertainty": norm_u, "label": -1})  # OOD 用 -1 占位

        df_un = pd.DataFrame(data_list)

        df_un = df_un.sort_values(by="type", ascending=True).reset_index(drop=True)

        # 获取索引
        x = np.arange(len(df_un))
        y = df_un["uncertainty"].values
        types = df_un["type"].values

        # 分离 ID / OOD 样本
        id_mask = types == "id"
        ood_mask = types == "ood"

        num_id = id_mask.sum()
        num_ood = ood_mask.sum()
        plt.rcParams.update({'font.size': 12})
        # 绘图
        plt.figure(figsize=(6, 5))

        # 散点图
        plt.scatter(x[id_mask], y[id_mask], s=10, color="blue", label="ID Samples")
        plt.scatter(x[ood_mask], y[ood_mask], s=10, color="red", label="OOD Samples")

        # 分界线
        plt.axvline(num_id, color="black", linestyle="--")

        # 添加箭头标注区域
        y_top = y.max() * 1.05
        plt.annotate('', xy=(num_id - 10, y_top), xytext=(10, y_top),
                     arrowprops=dict(arrowstyle='<->', color='black'))
        plt.annotate('', xy=(num_id + num_ood - 10, y_top), xytext=(num_id + 10, y_top),
                     arrowprops=dict(arrowstyle='<->', color='black'))

        # 文字标签
        plt.text(num_id / 2, y_top * 1.03, 'ID Samples', ha='center', fontsize=12)
        plt.text(num_id + num_ood / 2, y_top * 1.03, 'OOD Samples', ha='center', fontsize=12)

        # 坐标轴设置
        plt.xlabel('Sample Number')
        plt.ylabel('Normalized Uncertainty')
        plt.ylim(0, y_top * 1.03)
        plt.xlim(0, num_id + num_ood)
        plt.box(False)
        plt.tight_layout()
        # plt.show()
        if args.post == True:
            savename = os.path.join(args.save_dir, 'plot_post_ood_point.png')
        else:
            savename = os.path.join(args.save_dir, 'plot_ood_point.png')

            # plt.savefig(savename, format='png')
        plt.savefig(savename, dpi=300, bbox_inches="tight", format='png')
        plt.close()

    def plot_in_uncertainty_distribution(self):
        args = self.args
        if args.post == True:
            test_path = os.path.join(args.save_dir, "test_post.npz")
            ood_path = os.path.join(args.save_dir, "ood_post.npz")
        else:
            test_path = os.path.join(args.save_dir, "test.npz")
            ood_path = os.path.join(args.save_dir, "ood.npz")



        test_data = np.load(test_path)
        ood_data = np.load(ood_path)

        uncertainty_id = test_data["uncertainty"].flatten()
        labels_id = test_data["label"].flatten()
        uncertainty_ood = ood_data["uncertainty"].flatten()

        # print(len(uncertainty_id))
        # print(len(uncertainty_ood))

        all_uncertainty = np.concatenate([uncertainty_id, uncertainty_ood])
        min_u, max_u = all_uncertainty.min(), all_uncertainty.max()

        data_list = []
        for u, l in zip(uncertainty_id, labels_id):
            norm_u = (u - min_u) / (max_u - min_u)
            data_list.append({"type": "id", "uncertainty": norm_u, "label": int(l)})
        for u in uncertainty_ood:
            norm_u = (u - min_u) / (max_u - min_u)
            data_list.append({"type": "ood", "uncertainty": norm_u, "label": -1})  # OOD 用 -1 占位

        df_un = pd.DataFrame(data_list)
        # plt.figure(figsize=(4, 6))

        # ------------------ 图2: 各类别不确定性分布 (ID) ------------------
        g = sns.displot(df_un[df_un["type"] == "id"], x="uncertainty", hue="label", kind="kde", fill=True,
                        palette="tab10", height=4,  # figure height in inches
                        aspect=1.2,  legend=True  )
        # 获取 legend 对象并调整
        legend = g._legend
        legend.set_bbox_to_anchor((0.5, 0.75))

        # 字体稍大
        legend.set_title("Type")
        legend.get_title().set_fontsize(14)
        for text in legend.texts:
            text.set_fontsize(14)
        plt.xlabel("Normalized Uncertainty")
        plt.ylabel("Density")
        plt.subplots_adjust(right=0.95)

        new_labels = ["Majority", "Minority 1", "Minority 2", "Minority 3", "Minority 4"]
        for t, l in zip(g._legend.texts, new_labels):
            t.set_text(l)

        for ax in g.axes.flatten():
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel(ax.get_xlabel(), fontsize=14)
            ax.set_ylabel(ax.get_ylabel(), fontsize=14)



        if args.post == True:
            savename2 = os.path.join(args.save_dir, "uncertainty_per_class_post.png")
        else:
            savename2 = os.path.join(args.save_dir, "uncertainty_per_class.png")
        plt.savefig(savename2, dpi=300, bbox_inches="tight")
        plt.close()
        # plt.show()

        # ------------------ 图3: 各类别平均不确定性 ------------------
        mean_unc = (
            df_un[df_un["type"] == "id"]
            .groupby("label")["uncertainty"]
            .mean()
            .sort_index()
        )

        plt.figure(figsize=(8, 4))
        mean_unc.plot(kind="bar", width=0.8)
        plt.title("Mean Uncertainty per Class (ID)")
        plt.xlabel("Class Label")
        plt.ylabel("Mean Normalized Uncertainty")
        plt.tight_layout()
        savename3 = os.path.join(args.save_dir, "mean_uncertainty_per_class.png")
        plt.savefig(savename3, dpi=300, bbox_inches="tight")
        plt.close()
