import collections
import glob
import json
import math
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CSVDataset(Dataset):
    def __init__(self, file_path, label_name='label', transform=None):
        """
        :param file_path: Path to the CSV file (with header).
        :param label_name: The column name that contains the label.
        :param transform: Optional transform or pre-processing.
        """
        self.file_path = file_path
        self.label_name = label_name
        self.transform = transform

        # Read CSV header
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.column_names = reader.fieldnames  # list of columns
            if self.label_name not in self.column_names:
                raise ValueError(f"Label column '{self.label_name}' not found in {self.file_path}")

        # Store rows in memory
        self.data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

        # Pre-convert numeric columns if desired
        # (Optional: you can do this in __getitem__ instead)
        # Example: turn each field into float except maybe the label
        self.numeric_columns = [col for col in self.column_names if col != self.label_name]
        for i, row in enumerate(self.data):
            for col in self.numeric_columns:
                self.data[i][col] = float(self.data[i][col])
            # The label is typically int or float:
            self.data[i][self.label_name] = float(self.data[i][self.label_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        label = row[self.label_name]

        # Build feature dict (excluding label column).
        features = {}
        for col in self.numeric_columns:
            features[col] = row[col]

        # Convert to PyTorch tensors
        # For example, if each feature is scalar, we might keep them as float.
        # If you prefer all features in one tensor, you can concatenate them:
        #     x = torch.tensor([features[col] for col in self.numeric_columns], dtype=torch.float32)
        # But we’ll keep a dict for demonstration.

        feature_tensors = {}
        for key, val in features.items():
            feature_tensors[key] = torch.tensor([val], dtype=torch.float32)
            # Using shape [1] to mimic the "expand_dim" from TF

        label_tensor = torch.tensor([label], dtype=torch.float32)  # shape [1], like expand_dim

        if self.transform:
            feature_tensors, label_tensor = self.transform(feature_tensors, label_tensor)

        return feature_tensors, label_tensor


def get_dataset(file_path,
                label_name='label',
                shuffle=True,
                batch_size=32,
                shuffle_seed=None,
                num_workers=0):
    """
    :param file_path: Path to CSV.
    :param label_name: Name of label column.
    :param shuffle: Whether to shuffle the dataset.
    :param batch_size: Batch size.
    :param shuffle_seed: If set, use that seed.
    :param num_workers: Number of workers for parallel data loading.
    :return: (data_loader, steps_per_epoch, n_data)
    """
    # Create dataset
    dataset = CSVDataset(file_path, label_name=label_name)

    n_data = len(dataset)
    steps_per_epoch = int(np.ceil(n_data / float(batch_size)))

    # If you want reproducible shuffling, set the generator’s seed
    generator = None
    if shuffle_seed is not None:
        g = torch.Generator()
        g.manual_seed(shuffle_seed)
        generator = g

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             generator=generator,
                             drop_last=False)
    return data_loader, steps_per_epoch, n_data


class MultiDomainDataset(object):
    def __init__(self, conf):
        self.dataset_path = conf['dataset_path']
        self.domain_split_path = osp.join(self.dataset_path, conf['domain_split_path'])
        self.seed = conf['seed']
        self.conf = conf
        self.batch_size = self.conf['batch_size']
        # Buffer size in TF doesn't directly translate to PyTorch,
        # but you can approximate by using shuffle + num_workers.
        self.shuffle_buffer_size = self.conf.get('shuffle_buffer_size', 10000)
        self.num_workers = self.conf.get('num_parallel_reads', 0)  # PyTorch setting

        # Load user/item counts
        with open(osp.join(self.domain_split_path, "processed_data/uid2id.json"), "r") as f:
            raw2id = json.load(f)
            self.n_uid = raw2id['id']
        with open(osp.join(self.domain_split_path, "processed_data/pid2id.json"), "r") as f:
            raw2id = json.load(f)
            self.n_pid = raw2id['id']

        if conf['name'] == "Taobao":
            with open(osp.join(self.domain_split_path, "processed_data/item_emb.json"), "r") as f:
                self.item_emb = json.load(f)
            with open(osp.join(self.domain_split_path, "processed_data/user_emb.json"), "r") as f:
                self.user_emb = json.load(f)

        domains_list = glob.glob(osp.join(self.domain_split_path, "domain_*"))
        domains_list.sort(key=lambda x: int(x.split("_")[-1]))
        self.n_domain = len(domains_list)
        print(f"Found {self.n_domain} domain(s) in: {self.domain_split_path}")

        self.train_dataset = collections.OrderedDict()
        self.val_dataset = collections.OrderedDict()
        self.test_dataset = collections.OrderedDict()
        self.ctr_ratio = collections.OrderedDict()

        for d_path in tqdm(domains_list):
            domain_name = osp.split(d_path)[-1]
            domain_idx = int(domain_name.split("_")[-1])

            # By default train data is shuffled; you might override if needed
            shuffle_train = not bool(self.conf.get('fixed_train', False))

            # TRAIN
            train_loader, n_train_step, n_train = get_dataset(
                osp.join(d_path, "train.csv"),
                label_name='label',
                shuffle=shuffle_train,
                batch_size=self.batch_size,
                shuffle_seed=self.seed,
                num_workers=self.num_workers
            )
            # VAL
            val_loader, n_val_step, n_val = get_dataset(
                osp.join(d_path, "val.csv"),
                label_name='label',
                shuffle=False,
                batch_size=self.batch_size,
                shuffle_seed=self.seed,
                num_workers=self.num_workers
            )
            # TEST
            test_loader, n_test_step, n_test = get_dataset(
                osp.join(d_path, "test.csv"),
                label_name='label',
                shuffle=False,
                batch_size=self.batch_size,
                shuffle_seed=self.seed,
                num_workers=self.num_workers
            )

            # Domain property
            with open(osp.join(d_path, "domain_property.json"), 'r') as f:
                domain_property = json.load(f)
            self.ctr_ratio[domain_idx] = domain_property['ctr_ratio']

            self.train_dataset[domain_idx] = {
                "data": train_loader,
                "n_step": n_train_step,
                "n_data": n_train
            }
            self.val_dataset[domain_idx] = {
                "data": val_loader,
                "n_step": n_val_step,
                "n_data": n_val
            }
            self.test_dataset[domain_idx] = {
                "data": test_loader,
                "n_step": n_test_step,
                "n_data": n_test
            }

    def get_train_dataset(self, domain_idx):
        """
        :return: Dict with "data" (DataLoader), "n_step", "n_data".
        """
        return self.train_dataset[domain_idx]

    def get_val_dataset(self, domain_idx):
        return self.val_dataset[domain_idx]

    def get_test_dataset(self, domain_idx):
        return self.test_dataset[domain_idx]

    @property
    def dataset_info(self):
        total_train, total_val, total_test = 0, 0, 0
        info = {
            'n_user': self.n_uid,
            'n_item': self.n_pid
        }
        for i in self.train_dataset:
            info[i] = {
                "n_train": self.train_dataset[i]['n_data'],
                "n_val": self.val_dataset[i]['n_data'],
                "n_test": self.test_dataset[i]['n_data'],
                "ctr_ratio": self.ctr_ratio[i]
            }
            total_train += self.train_dataset[i]['n_data']
            total_val += self.val_dataset[i]['n_data']
            total_test += self.test_dataset[i]['n_data']
        info["total_train"] = total_train
        info['total_val'] = total_val
        info['total_test'] = total_test
        return info
