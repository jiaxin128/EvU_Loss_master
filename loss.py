import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc


class A2EvULoss(nn.Module):

    def __init__(self):
        super(A2EvULoss, self).__init__()
        self.eps = 1e-10

    def forward(self, output, target, num_classes):

        evidence = exp_evidence(output)
        alpha = evidence + 1
        max_alpha, predictions = torch.max(alpha, 1)
        unc = num_classes / torch.sum(alpha, dim=1)

        th_list = np.linspace(0, 1, 21)
        umin = torch.min(unc)
        umax = torch.max(unc)
        evu_list = []
        unc_list = []

        auc_evu = torch.ones(1, device=output.device)
        auc_evu.requires_grad_(True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t, device=output.device) * (umax - umin))
            n_ac = torch.zeros(1, device=output.device)
            n_ic = torch.zeros(1, device=output.device)
            n_au = torch.zeros(1, device=output.device)
            n_iu = torch.zeros(1, device=output.device)

            for i in range(len(target)):
                if ((target[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    n_ac += max_alpha[i] * (1 - torch.tanh(unc[i]))
                elif ((target[i].item() == predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    n_au += max_alpha[i] * torch.tanh(unc[i])
                elif ((target[i].item() != predictions[i].item())
                      and unc[i].item() <= unc_th.item()):
                    n_ic += (1 - max_alpha[i]) * (1 - torch.tanh(unc[i]))
                elif ((target[i].item() != predictions[i].item())
                      and unc[i].item() > unc_th.item()):
                    n_iu += (1 - max_alpha[i]) * torch.tanh(unc[i])

            evu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            evu_list.append(evu.detach().cpu().numpy())
            unc_list.append(unc_th)

        auc_value = auc(th_list, evu_list)
        auc_evu = torch.tensor(auc_value, device=output.device, dtype=torch.float32)
        auc_loss = -1 * torch.log(auc_evu + self.eps)
        return auc_loss


class EvULoss(nn.Module):

    def __init__(self):
        super(EvULoss, self).__init__()
        self.eps = 1e-10

    def forward(self, output, target, optimal_uncertainty_threshold, num_classes):

        evidence = exp_evidence(output)
        alpha = evidence + 1
        max_alpha, predictions = torch.max(alpha, 1)
        unc = num_classes / torch.sum(alpha, dim=1)
        umin = torch.min(unc)
        umax = torch.max(unc)
        unc_th = umin + (torch.tensor(optimal_uncertainty_threshold, device=output.device) * (umax - umin))

        n_ac = torch.zeros(1, device=output.device)
        n_ic = torch.zeros(1, device=output.device)
        n_au = torch.zeros(1, device=output.device)
        n_iu = torch.zeros(1, device=output.device)

        for i in range(len(target)):
            if ((target[i].item() == predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                n_ac += max_alpha[i] * (1 - torch.tanh(unc[i]))
            elif ((target[i].item() == predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                n_au += max_alpha[i] * torch.tanh(unc[i])
            elif ((target[i].item() != predictions[i].item())
                  and unc[i].item() <= unc_th.item()):
                n_ic += (1 - max_alpha[i]) * (1 - torch.tanh(unc[i]))
            elif ((target[i].item() != predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                n_iu += (1 - max_alpha[i]) * torch.tanh(unc[i])

        evu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        evu_loss = -1 * self.beta * torch.log(evu + self.eps)
        return evu_loss


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(y)


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = exp_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss
