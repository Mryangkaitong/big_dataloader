# -*- coding: utf-8 -*-
# @Author  : kaitongyang
# @Email   : 1500936094@qq.com
# @Software: PyCharm

import math
import json
import pickle
import time
import numpy as np

from pbatch import batch
from sampler import RandomSampler
from sampler import SequentialSampler
from sampler import SortedSampler
from multiprocessing import Process
from multiprocessing import Process, Manager, Lock

class DataLoader(object):
    
    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--init_read_count", type=int, default=0)
        group.add_argument("--instances_buffer_size", type=int, default=25600)
        group.add_argument("--dataloder_num_process", type=int, default=-1)
        return group

    def __init__(self, hparams, dataset_path=None, dataset_idx_path=None, collate_fn=None, proc_id=-1, proc_num=-1, is_train=True, dataset=None):
        self.collate_fn = collate_fn

        self.dataset_path = dataset_path
        self.dataset_idx_path = dataset_idx_path
        self.is_train = is_train

        if is_train:
            self.instances_buffer_size = hparams.instances_buffer_size
            self.batch_size = hparams.train_batch_size

            self.num_process = hparams.dataloder_num_process
            self.proc_id = proc_id
            self.proc_num = proc_num
            manager = Manager()
            self.buffer_single = manager.list()
            self.buffer_batch = manager.list()
            self.buffer_lock = Lock()
            self.buffer_thread = None
            self.read_count_global = manager.Value('i', 0)
        else:
            self.dataset = dataset
            sampler = SequentialSampler(dataset)
            def reader():
                for idx in sampler:
                    yield idx
            self.reader = batch(reader, batch_size=hparams.vaild_batch_size, drop_last=False)
            self.num_batches = math.ceil(len(dataset) / hparams.vaild_batch_size)

    def __len__(self):
        if self.is_train:
            with open(self.dataset_idx_path, 'rb') as f:
                dataset_idx_list = pickle.load(f)
                dataset_idx_list = dataset_idx_list[self.proc_id:: self.proc_num]
            return len(dataset_idx_list) // self.batch_size
        else:
            return self.num_batches

    def set_start_idx(self, idx):
        self.read_count_global.value = idx

    def get_current_read_count(self):
        return self.read_count_global.value

    def _fill_buf(self):
        if self.buffer_thread is None:
            self.buffer_thread = []
            for process in range(self.num_process):
                buffer_thread = Process(target=self.buf_thread, args=(process, self.num_process))
                buffer_thread.start()
                self.buffer_thread.append(buffer_thread)

    def buf_thread(self, process, num_process):
        print('=========start buf thread=========')
        read_count = self.read_count_global.value
        while True:
            with open(self.dataset_idx_path, 'rb') as f:
                dataset_idx_list = pickle.load(f)
                print('loaded dataset idx', dataset_idx_list[:10])
                dataset_idx_list = dataset_idx_list[self.proc_id:: self.proc_num]
            print('buffer in: ', process, num_process)
            f_read = open(self.dataset_path, "rb")
            num_data = len(dataset_idx_list)
            while True:
                # self.buffer_single装满啦，等等训练取走一些数据再装
                if len(self.buffer_single) >= self.instances_buffer_size:
                    max_batch_buffer = max(self.instances_buffer_size / self.batch_size, 256)
                    if len(self.buffer_batch) < max_batch_buffer:
                        self._fill_batch()
                    else:
                        time.sleep(0.1)
                        continue
                # 装到当前epcoh最后一个数据了
                if read_count >= num_data:
                    break
                start_idx = dataset_idx_list[read_count]
                read_count += 1
                # 多个进程互不影响，确保不重复装数据
                if read_count % num_process != process:  # skip
                    continue
                self.read_count_global.value = read_count
                # 一个个装对应位置的数据
                f_read.seek(start_idx, 0)
                self.buffer_single.append(json.loads(f_read.readline().strip()))
            f_read.close()
            read_count = 0

    def _fill_batch(self):
        while len(self.buffer_single) > self.instances_buffer_size * 0.75:
            self.buffer_lock.acquire()
            num_data = len(self.buffer_single)
            batch_idx = np.random.choice(num_data, self.batch_size, replace=num_data < self.batch_size)
            batch_idx.sort()
            instances = [self.buffer_single.pop(i) for i in batch_idx[::-1]]
            self.buffer_lock.release()
            self.buffer_batch.append(self.collate_fn(instances))

    def __iter__(self):
        if self.is_train:
            if self.buffer_thread is None:
                self._fill_buf()
            while True:
                if len(self.buffer_batch):
                    yield self.buffer_batch.pop(0)
        else:
            for batch_indices in self.reader():
                samples = [self.dataset[idx] for idx in batch_indices]
                yield self.collate_fn(samples)
