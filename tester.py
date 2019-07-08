import numpy as np
import pdb
import os

from torch.utils.data import DataLoader
from torch import nn
import torch

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall, MetricsLambda
from ignite.contrib.handlers import tqdm_logger

import env
from network import CNN_IMU
from windataset import windowDataSet
from netevaluator import GaussianNoise



class Tester:
    
    def __init__(self, type):

        print('[Tester] - Initializing tester')
        self.type = type
        if type == 0:
            self.modelurl = env.models_url
            self.dataurl = env.window_url
        elif type == 1:
            self.modelurl = env.marker_models_url
            self.dataurl = env.marker_window_url
        elif type == 2:
            self.modelurl = env.cross_models_url
            self.dataurl = env.window_url
        elif type == 3:
            self.modelurl = env.cross_marker_models_url
            self.dataurl = env.marker_window_url


    def filterStateDict(self, model_path, net):
        print(net.state_dict()['fc2.weight'])
        
        currdict = net.state_dict()

        for k, v in net.state_dict().items():
            print(k)
        
        print('////////')

        nnmodel = torch.load(model_path)
        for k, v in nnmodel.items():
            if 'fc' in k:
                print(k)
                currdict.update({k: v})

        net.load_state_dict(currdict)

        print('//////')
        print(nnmodel['fc2.weight'])
        print('///////')
        print(net.state_dict()['fc2.weight'])

        
    def load_checkpoint(self, config):
        # {window length} - {window step} - {Learning rate}
        
        keystr = "{}_{}_{}".format(config['win_len'],config['win_step'],config['lr'])
        
        saved_model_name = [n for n in os.listdir(self.modelurl) if keystr in n][0]
        saved_model_path = os.path.join(self.modelurl, saved_model_name)
        print(saved_model_name)
        device = torch.device(
            config['gpucore'] if torch.cuda.is_available() else "cpu")
        net = CNN_IMU(config)
        self.filterStateDict(saved_model_path, net)
        net.load_state_dict(torch.load(saved_model_path))
        net = net.to(device)
        
        return net, device
    
    
    def get_data_loader(self, config):

        train_batch_size = config['batch_validate']

        # OBTAINING TEST - DATASET / DATALOADER
        test_win_dir = self.dataurl.format(config['win_len'],
                                           config['win_step'],
                                           'test')
        test_dataset = windowDataSet(dir=test_win_dir,
                                     transform=GaussianNoise(0, 1e-2, self.type))

        test_loader = DataLoader(test_dataset,
                                 batch_size=train_batch_size,
                                 shuffle=False,
                                 num_workers=4)


        return test_loader, test_dataset.__len__()

    
    def F1(self, precision, recall):
        return (precision * recall * 2 / (precision + recall + 1e-20)).mean()
    
    
    def get_metrics(self):
        
        criterion = nn.CrossEntropyLoss()
        
        precision = Precision(average=False)
        recall = Recall(average=False)

        metrics = {
            'accuracy': Accuracy(),
            'loss': Loss(criterion),
            'precision': precision,
            'recall': recall,
            'f1': MetricsLambda(self.F1, precision, recall)
        }
        
        return metrics
    
    
    def create_supervisor(self, config):
        
        # GET LOADED NETWORK
        net, device = self.load_checkpoint(config)
        
        # SET METRICS
        metrics = self.get_metrics()
        
        # CREATE IGNITE TESTER
        tester = create_supervised_evaluator(net,
                                             metrics=metrics,
                                             device=device)
        
        # TQDM OBSERVERS
        pbar = tqdm_logger.ProgressBar()
        pbar.attach(tester)
        
        @tester.on(Events.EPOCH_COMPLETED)
        def log_test_results(engine):
            m = engine.state.metrics
            print('Test results for WinSize [{}], WinStep [{}], LR [{}]'.format(
                                                                          config['win_len'],
                                                                          config['win_step'],
                                                                          config['lr']))
            print('Loss:    {}\nAccuracy    {}\nF1:    {}'.format(
                                                                  m['loss'],
                                                                  m['accuracy'],
                                                                  m['f1']))
        return tester
        
        
    def runTest(self, config):
        # GET DATA AND TESTER
        test_loader, test_len = self.get_data_loader(config)
        tester = self.create_supervisor(config)
        
        # RUN TEST
        tester.run(test_loader)
       
        return tester.state.metrics

#
#
# class GaussianNoise(object):
#     """
#     Add Gaussian noise to a window data sample
#     """
#
#     def __init__(self, mu, sigma):
#         self.mu = mu
#         self.sigma = sigma
#
#     def __call__(self, sample):
#         data = sample['data']
#         label = np.long(sample['label'])
#         data += np.random.normal(self.mu,
#                                  self.sigma,
#                                  data.shape)
#         data = np.expand_dims(data, 0)
#         return (data, label)
#
#
#
#
        
        
        
        
        
        
