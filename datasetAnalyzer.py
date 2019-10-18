from ignite.utils import to_onehot
import pandas as pd
import numpy as np
import torch
import glob
import env
import tqdm
import pdb


class Analyzer:

    def __init__(self):
        # self.markers_folder = "/Users/dramirez.c90/Desktop/dataset/m/"
        self.data_folder = env.dataset_url
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else "cpu")
        self.sum = torch.zeros(7).type(torch.IntTensor)

    def remove_class(self, data, cl):

        indices = data != cl

        return data[indices]

    def read_data(self, path):
        data = pd.read_csv(path)
        data = data.iloc[:, 1]
        data = self.remove_class(data, 7)

        if (data.size > 0):
            data = torch.tensor(data.values)
            data2 = to_onehot(data, 7)
            t_sum = data2.sum(dim=0).type(torch.IntTensor)
            self.sum = self.sum + t_sum

    def run(self, input):
        data_dict = input['data_sets']
        name = input['name']
        for dir in data_dict:
            files = glob.glob(self.data_folder.format(dir))
            for i, f in enumerate(tqdm.tqdm(files)):
                # print('Running for {}'.format(f))
                self.read_data(f)
        print('Results for {}'.format(name))
        print(self.sum)

class StatAnalyzer:

    def __init__(self):
        self.list = [0,1,2,3,4,5,6]

    def read_data(self, path):
        data = pd.read_csv(path)
        data = data.iloc[:, 1]
        data = self.remove_class(data, 7)

        if (data.size > 0):
            a=0

    def avg(self, x):
        print(x)
        print('log')

    def run(self):
        keys = set(self.list)
        class_dict = dict((x, list()) for x in keys)
        print(class_dict)
        prev = 999
        count = 0
        for f_class in self.list:
            if f_class == prev:
                count+=1
            else:
                if prev != 999:
                    class_dict[prev].append(count)
                count = 1
                prev = f_class
        class_dict[prev].append(count)
        print(class_dict)
        stats = dict((x, self.avg(class_dict[x])) for x in class_dict.keys())
        return class_dict


if __name__ == '__main__':
    # an = Analyzer()
    #
    # an.run({'name': 'Training', 'data_sets': ['P01', 'P02', 'P03']})
    # an.run({'name': 'Validating', 'data_sets': ['P04']})
    # an.run({'name': 'Testing', 'data_sets': ['P05', 'P06']})


    a = StatAnalyzer()

    stats = a.run()
    print(stats)


    # /Users/dramirez.c90/Desktop/dataset/m/Subject01_01.csv
