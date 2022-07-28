import pickle
import numpy as np


class Cifar100(object):
    def __init__(self, num_task=5, random_flag=True):
        super(Cifar100, self).__init__()

        with open("/home/luojing/datasets/CIFAR100/cifar-100-python/train", "rb") as f:
            self.train = pickle.load(f, encoding="latin1")

        with open("/home/luojing/datasets/CIFAR100/cifar-100-python/test", "rb") as f:
            self.test = pickle.load(f, encoding="latin1")

        self.train_data = self.train["data"]
        self.train_labels = self.train["fine_labels"]

        self.test_data = self.test["data"]
        self.test_labels = self.test["fine_labels"]

        self.num_task = num_task
        self.random_flag = random_flag
        self.current_step = 0

        self._initialize()

    def _initialize(self):
        self.train_groups = [[] for i in range(self.num_task)]
        self.test_groups = [[] for i in range(self.num_task)]

        labels_map = {}

        if self.random_flag:
            np.random.seed(1)
            labels_map_v = np.random.permutation(100)            
        else:
            labels_map_v = np.arange(100)

        for i in range(len(labels_map_v)):
            labels_map[labels_map_v[i]] = i

        assert 100 % self.num_task == 0
        label_step = 100 // self.num_task

        for train_data, train_label in zip(self.train_data, self.train_labels):
            train_data_r = train_data[:1024].reshape(32, 32)
            train_data_g = train_data[1024:2048].reshape(32, 32)
            train_data_b = train_data[2048:].reshape(32, 32)
            train_data = np.dstack((train_data_r, train_data_g, train_data_b))

            for i in range(self.num_task):
                if label_step * i <= labels_map[train_label] < label_step * (i + 1):
                    self.train_groups[i].append((train_data, labels_map[train_label]))

        for test_data, test_label in zip(self.test_data, self.test_labels):
            test_data_r = test_data[:1024].reshape(32, 32)
            test_data_g = test_data[1024:2048].reshape(32, 32)
            test_data_b = test_data[2048:].reshape(32, 32)
            test_data = np.dstack((test_data_r, test_data_g, test_data_b))

            for i in range(self.num_task):
                if label_step * i <= labels_map[test_label] < label_step * (i + 1):
                    self.test_groups[i].append((test_data, labels_map[test_label]))
            
    def get_next_classes(self, step):
        assert 0 <= step < self.num_task
        return self.train_groups[step], self.test_groups[step]
