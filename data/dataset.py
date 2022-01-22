# -*- coding: utf-8 -*-
# @Author  : kaitongyang
# @Email   : 1500936094@qq.com
# @Software: PyCharm

import pickle
import json

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class Dataset(object):
    """ Basic Dataset interface class. """

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Dataset")
        group.add_argument("--data_dir", type=str, required=True,
                           help="The dataset dir.")
        group.add_argument("--use_record_idx", type=str2bool, default=False,
                            help="Whether to use record_idx.")
        group.add_argument("--dataset_idx_path", type=str, default=None,
                           help="dataset_idx path.")
        group.add_argument("--data_type", type=str, required=True,
                           choices=["multi", "multi_knowledge"],
                           help="The type of dataset.")
        return group

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyDataset(Dataset):
    """
    Lazy load dataset from disk.

    Each line of data file is a preprocessed example.
    """

    def __init__(self, data_file, use_record_idx=False, dataset_idx_path=None, transform=lambda s: json.loads(s)):
        """
        Initialize lazy dataset.

        By default, loading .jsonl format.

        :param data_file
        :type str

        :param transform
        :type callable
        """
        self.data_file = data_file
        self.transform = transform
        if use_record_idx:
            with open(dataset_idx_path, 'rb') as f:
                print('loading dataset idx: {}'.format(dataset_idx_path))
                self.offsets = pickle.load(f)
                print('loaded dataset idx', self.offsets[:10])
        else:
            self.offsets = [0]
            with open(data_file, "r", encoding="utf-8") as fp:
                while fp.readline() != "":
                    self.offsets.append(fp.tell())
            self.offsets.pop()
        self.fp = open(data_file, "r", encoding="utf-8")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self.fp.seek(self.offsets[idx], 0)
        return self.transform(self.fp.readline().strip())
