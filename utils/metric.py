import numpy as np
import pandas as pd
from sklearn.metrics import auc

def AUC_evidence_vs_uncertainty(output, target, uncertainty):

    umin = np.min(uncertainty)
    umax = np.max(uncertainty)

    th_list = np.linspace(0, 1, 21)
    evu_list = []
    unc_list = []

    for t in th_list:
        n_ac = 0
        n_ic = 0
        n_au = 0
        n_iu = 0
        unc_th = umin + (t * (umax - umin))
        for i in range(len(target)):
            if ((target[i].item() == output[i].item())
                    and uncertainty[i].item() <= unc_th):
                """ accurate and certain """
                n_ac += 1
            elif ((target[i].item() == output[i].item())
                  and uncertainty[i].item() > unc_th):
                """ accurate and uncertain """
                n_au += 1
            elif ((target[i].item() != output[i].item())
                  and uncertainty[i].item() <= unc_th):
                """ inaccurate and certain """
                n_ic += 1
            elif ((target[i].item() != output[i].item())
                  and uncertainty[i].item() > unc_th):
                """ inaccurate and uncertain """
                n_iu += 1


        evu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
        evu_list.append(evu)
        unc_list.append(unc_th)

    auc_evu = auc(th_list, evu_list)
    return auc_evu


def evidence_vs_uncertainty(output, target, uncertainty, optimal_threshold):
    n_ac = 0
    n_ic = 0
    n_au = 0
    n_iu = 0

    umin = np.min(uncertainty)
    umax = np.max(uncertainty)

    unc_th = umin + (optimal_threshold * (umax - umin))

    for i in range(len(target)):
        if ((target[i].item() == output[i].item())
                and uncertainty[i].item() <= unc_th):
            """ accurate and certain """
            n_ac += 1
        elif ((target[i].item() == output[i].item())
              and uncertainty[i].item() > unc_th):
            """ accurate and uncertain """
            n_au += 1
        elif ((target[i].item() != output[i].item())
              and uncertainty[i].item() <= unc_th):
            """ inaccurate and certain """
            n_ic += 1
        elif ((target[i].item() != output[i].item())
              and uncertainty[i].item() > unc_th):
            """ inaccurate and uncertain """
            n_iu += 1

    # print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ', n_iu)
    evu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
    pac = n_ac / (n_ac + n_ic + 0.001)
    pui = n_iu / (n_ic + n_iu + 0.001)

    return pac, pui, evu


def compute_calibration(pred_labels, confidences, true_labels, uncertainty, num_bins=10):
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (len(uncertainty) == len(pred_labels))
    assert (len(uncertainty) == len(true_labels))
    assert (num_bins > 0)

    """ Metric 1 """

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float32)
    bin_errors = np.zeros(num_bins, dtype=np.float32)
    bin_confidences = np.zeros(num_bins, dtype=np.float32)
    bin_uncertainty = np.zeros(num_bins, dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_errors[b] = np.mean(true_labels[selected] != pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_uncertainty[b] = np.mean(uncertainty[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    un_gaps = np.abs(bin_errors - bin_uncertainty)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    uce = np.sum(un_gaps * bin_counts) / np.sum(bin_counts)

    return avg_acc, ece, uce