import torch.nn.functional as F
import torch.nn as nn
import pdb
import numpy as np


class CNN_IMU(nn.Module):
    """
        My own CNN_IMU implementation, extends nn.Module from tensorflow
    """
    def __init__(self, config):
        super(CNN_IMU, self).__init__()

        self.conv1 = nn.Conv2d(config['depth'],
                               config['n_filters'],
                               config['f_size'],
                               stride=(1,)*config['f_size'].__len__(),
                               padding=(0,)*config['f_size'].__len__())
        out_dim = (config['win_len']-4)/2
        self.conv2 = nn.Conv2d(config['n_filters'], config['n_filters'], (5,1))
        out_dim = (out_dim-4)/2
        self.fc1 = nn.Linear(int(out_dim)*config['n_filters']*config['channels'], 512)
        self.fc2 = nn.Linear(512, config['n_classes'])


    def forward(self, x):
        """
        :param x: Input to perform forward pass
        :return: Output of network forward pass
        """
        # pdb.set_trace()
        x = x.float()
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,1))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


    def num_flat_features(self, x):
        """
        :param x: Multidim tensor to flatten
        :return: num_features: flattened features of tensor x
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

