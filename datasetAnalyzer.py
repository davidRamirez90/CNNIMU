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
        self.data_dict = ['P01', 'P02', 'P03']
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

    def run(self):
        for dir in self.data_dict:
            files = glob.glob(self.data_folder.format(dir))
            for i, f in enumerate(tqdm.tqdm(files)):
                print('Running for {}'.format(f))
                self.read_data(f)

        print(self.sum)



if __name__ == '__main__':
    an = Analyzer()
    an.run()



    # /Users/dramirez.c90/Desktop/dataset/m/Subject01_01.csv