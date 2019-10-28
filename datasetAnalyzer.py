from ignite.utils import to_onehot
import pandas as pd
import numpy as np
import torch
import glob
import env
import tqdm
import csv
import pdb


class Analyzer:

    def __init__(self):
        # self.data_folder = "/Users/dramirez.c90/Desktop/dataset/m/"
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

    def reset(self):
        self.sum = torch.zeros(7).type(torch.IntTensor)

    def run(self, input):
        prefix = '/vol/corpora/har/DFG_Project/2019/MoCap/recordings_2019_06/09_New_Representation/{}'
        filenames = [
            'S01_P07_R01_A03_N01_norm.csv',
            'S01_P07_R02_A03_N01_norm.csv',
            'S01_P09_R01_A03_N01_norm.csv',
            'S01_P09_R02_A03_N01_norm.csv',
            'S01_P10_R01_A17_N01_norm.csv',
            'S01_P10_R02_A17_N01_norm.csv',
            'S01_P11_R01_A03_N01_norm.csv',
            'S01_P11_R02_A03_N01_norm.csv',
            'S01_P13_R01_A08_N01_norm.csv',
            'S01_P13_R02_A08_N01_norm.csv',
            'S01_P14_R01_A06_N01_norm.csv',
            'S01_P14_R02_A06_N01_norm.csv',
        ]
        data_dict = input['data_sets']
        name = input['name']
        # for dir in data_dict:
        #     files = glob.glob(self.data_folder.format(dir))
        #     for i, f in enumerate(tqdm.tqdm(files)):
        #         # print('Running for {}'.format(f))
        #         self.read_data(f)
        # print('Results for {}'.format(name))
        # print(self.sum)

        for i, f in enumerate(filenames):
            print('Reading {}'.format(f))
            self.read_data(prefix.format(f))
            print('Results for {}'.format(f))
            print(self.sum)
            self.reset()

class StatAnalyzer:

    def __init__(self):
        self.data_folder = env.dataset_url
        self.keys = [0, 1, 2, 3, 4, 5, 6, 7]
        self.class_dict = dict((x, list()) for x in self.keys)

    def remove_class(self, data, cl):
        indices = data != cl
        return data[indices]

    def read_data(self, path):
        data = pd.read_csv(path)
        data = data.iloc[:, 1]
        # data = self.remove_class(data, 7)
        prev = 999
        count = 0
        if (data.size > 0):
            for f_class in data:
                if f_class == prev:
                    count+=1
                else:
                    if prev != 999:
                        self.class_dict[prev].append(count)
                    count = 1
                    prev = f_class
            self.class_dict[prev].append(count)
            print(self.class_dict)


    def calculate(self, inputVect):
        pdb.set_trace()
        prev = 999
        count = 0
        if (inputVect.size > 0):
            for f_class in inputVect:
                # pdb.set_trace()
                if f_class == prev:
                    count += 1
                else:
                    print(f_class)
                    if prev != 999:
                        self.class_dict[prev].append(count)
                    count = 1
                    prev = f_class
            self.class_dict[prev].append(count)
            print(self.class_dict)


    def run(self, input):


        data_dict = input['data_sets']
        name = input['name']
        for dir in data_dict:
            files = glob.glob(self.data_folder.format(dir))
            for i, f in enumerate(tqdm.tqdm(files)):
                print('Running for {}'.format(f))
                self.read_data(f)
        res_frame = pd.concat([pd.DataFrame(x) for x in self.class_dict.values()], axis=1)

        export = res_frame.to_csv('samplefile.csv', index=None, header=True)
        print('Results for {}'.format(name))
        print(self.sum)
        return True



if __name__ == '__main__':
    an = Analyzer()

    an.run({'name': 'Training', 'data_sets': ['P01', 'P02', 'P03']})
    # an.run({'name': 'Validating', 'data_sets': ['P04']})
    # # an.run({'name': 'Testing', 'data_sets': ['P05', 'P06']})
    #
    # path = "/Users/dramirez.c90/Desktop/dataset/m/Subject01_01.csv"
    # data = pd.read_csv(path)


    # a = StatAnalyzer()

    # stats = a.run({'name': 'Training', 'data_sets': ['P01', 'P02', 'P03', 'P04']})


