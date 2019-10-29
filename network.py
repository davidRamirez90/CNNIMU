import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import pdb
import numpy as np


class CNN_IMU(nn.Module):
    """
        My own CNN_IMU implementation, extends nn.Module from tensorflow
    """
    def __init__(self, config):
        super(CNN_IMU, self).__init__()

        self.conv11 = nn.Conv2d(config['depth'],
                               config['n_filters'],
                               config['f_size'])
        self.conv12 = nn.Conv2d(config['n_filters'],
                                config['n_filters'],
                                config['f_size'])
        out_dim = (config['win_len'] - 8) / 2
        self.conv21 = nn.Conv2d(config['n_filters'],
                               config['n_filters'],
                               config['f_size'])
        self.conv22 = nn.Conv2d(config['n_filters'],
                                config['n_filters'],
                                config['f_size'])
        out_dim = (out_dim - 8) / 2
        self.fc1 = nn.Linear(int(out_dim)*config['n_filters']*config['channels'], 512)
        # self.fc12 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(512, config['n_classes'])

        self.drop = nn.Dropout()

        self.initializeWeights()


    def initializeWeights(self):
        """
        ORTHONORMAL WEIGHT INITIALIZATION
        """
        init.orthogonal_(self.conv11.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv12.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv21.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv22.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.fc1.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.fc12.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.fc2.weight, init.calculate_gain('relu'))



    def forward(self, x):
        """
        :param x: Input to perform forward pass
        :return: Output of network forward pass
        """
        x = x.float()
        x = F.relu(self.conv11(x))
        # x = F.max_pool2d(x, (2,1))
        x = F.max_pool2d(F.relu(self.conv12(x)), (2,1))
        # conv block 1
        x = F.relu(self.conv21(x))
        x = F.max_pool2d(F.relu(self.conv22(x)), (2,1))
        # conv block 2
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop(x)
        # flattened layer + dropout
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        # 512 layer + dropout
        # x = F.relu(self.fc12(x))
        # x = self.drop(x)
        # 128 layer + dropout
        x = self.fc2(x)
        # end

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

