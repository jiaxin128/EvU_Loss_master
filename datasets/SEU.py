import os

import pandas as pd
from tqdm import tqdm

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

signal_size = 1024

fault_name = ['Health_20_0.csv', 'outer_20_0.csv', 'inner_20_0.csv', 'ball_20_0.csv', 'comb_20_0.csv']
fault_test_name = ['Health_30_2.csv', 'outer_30_2.csv', 'inner_30_2.csv', 'ball_30_2.csv', 'comb_30_2.csv']
fault_ood_name = ['Chipped_20_0.csv', 'Miss_20_0.csv', 'Root_20_0.csv', 'Surface_20_0.csv']
label = [0, 1, 2, 3, 4]
ood_label = [5, 6, 7, 8]


def data_load(root, dataset_name, label):
    filename = os.path.join(root, dataset_name)
    if dataset_name == "ball_20_0.csv":
        df = pd.read_csv(filename, skiprows=16, delimiter=",")
    else:
        df = pd.read_csv(filename, skiprows=16, delimiter="\t")
    fl = df.iloc[:, 1]
    data_length = len(fl)
    # print(data_length)
    fl = np.array(fl).reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= data_length:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    return data, lab


def get_files(root, class_prior):
    data = []
    lab = []
    num = [int(p * 600) for p in class_prior]
    for i in tqdm(range(len(fault_name))):
        data1, lab1 = data_load(root, fault_name[i], label=label[i])
        data += data1[:num[i]]
        lab += lab1[:num[i]]
    return [data, lab]


def get_files_val(root):
    data = []
    lab = []
    num = 601
    for i in tqdm(range(len(fault_name))):  # 故障数据标记
        data1, lab1 = data_load(root, fault_name[i], label=label[i])
        data += data1[num:(num + 200)]
        lab += lab1[num:(num + 200)]
    return [data, lab]


def get_files_test(root):
    data = []
    lab = []
    num = 601
    for i in tqdm(range(len(fault_test_name))):  # 故障数据标记
        data1, lab1 = data_load(root, fault_test_name[i], label=label[i])
        data += data1[num:(num + 200)]
        lab += lab1[num:(num + 200)]
    return [data, lab]


def get_files_ood(root):
    data = []
    lab = []
    data1, lab1 = data_load(root, fault_ood_name[0], label=ood_label[0])
    data += data1[0:1000]
    lab += lab1[0:1000]
    return [data, lab]


def data_transforms(dataset_type="train", normalize_type="0-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normalize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()
        ]),
        'val': Compose([
            Reshape(),
            Normalize(normalize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


class Mechanical_datasets(object):
    num_classes = 5
    inputchannel = 1

    def __init__(self, data_dir, class_prior, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.class_prior = class_prior

    def data_preprare(self, dataset_name):
        if dataset_name == 'train':
            list_data_train = get_files(self.data_dir, self.class_prior)
            train_pd = pd.DataFrame({"data": list_data_train[0], "label": list_data_train[1]})
            train_dataset = dataset(list_data=train_pd, transform=data_transforms(dataset_type="train"))
            return train_dataset
        elif dataset_name == 'val':
            list_data_val = get_files_val(self.data_dir)
            val_pd = pd.DataFrame({"data": list_data_val[0], "label": list_data_val[1]})
            val_dataset = dataset(list_data=val_pd, transform=data_transforms(dataset_type="val"))
            return val_dataset
        elif dataset_name == 'test':
            list_data_test = get_files_test(self.data_dir)
            test_pd = pd.DataFrame({"data": list_data_test[0], "label": list_data_test[1]})
            test_dataset = dataset(list_data=test_pd, transform=data_transforms(dataset_type="val"))
            return test_dataset
        else:
            list_data_ood = get_files_ood(self.data_dir)
            ood_pd = pd.DataFrame({"data": list_data_ood[0], "label": list_data_ood[1]})
            ood_dataset = dataset(list_data=ood_pd, transform=data_transforms(dataset_type="val"))
            return ood_dataset










