import numpy as np
import pdb
import os

from torch.utils.data import DataLoader
from torch import nn
import torch

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall, MetricsLambda, ConfusionMatrix, Metric
from ignite.contrib.handlers import tqdm_logger
from ignite.exceptions import NotComputableError
from ignite.utils import to_onehot


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
            self.modelurl = env.accel_models_url
            self.dataurl = env.accel_window_url
        elif type == 3:
            self.modelurl = env.cross_models_url
            self.dataurl = env.window_url
        elif type == 4:
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

        
    def load_checkpoint(self, config, it):
        # {window length} - {window step} - {Learning rate}
        
        keystr = "[{}]-CNNIMU_{}_{}_{}".format(it, config['win_len'],config['win_step'],config['lr'])
        saved_model_name = [n for n in os.listdir(self.modelurl) if keystr in n][0]
        saved_model_path = os.path.join(self.modelurl, saved_model_name)
        print(saved_model_path)
        #pdb.set_trace()
        device = torch.device(
            config['gpucore'] if torch.cuda.is_available() else "cpu")
        net = CNN_IMU(config)
        # net.load_state_dict(torch.load(saved_model_path))
        # self.filterStateDict(saved_model_path, net)
        net.load_state_dict(torch.load(saved_model_path))
        net = net.to(device)

        print(net)
        print(device)
        
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
            'accPerClass': LabelwiseAccuracy(),
            'confMatrix': customConfusionMatrix(num_classes=7),
            'loss': Loss(criterion),
            'precision': precision,
            'recall': recall,
            'f1': MetricsLambda(self.F1, precision, recall)
        }
        
        return metrics
    
    
    def create_supervisor(self, config, it):
        
        # GET LOADED NETWORK
        net, device = self.load_checkpoint(config, it)
        
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

        print(tester)
        return tester
        
        
    def runTest(self, config, it):
        # GET DATA AND TESTER
        test_loader, test_len = self.get_data_loader(config)
        #pdb.set_trace()
        tester = self.create_supervisor(config, it)
        #pdb.set_trace()
        
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

class customConfusionMatrix(Metric):
    def __init__(self, num_classes, average=None, output_transform=lambda x: x):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of ['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(customConfusionMatrix, self).__init__(output_transform=output_transform)

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float)
        self._num_examples = 0

    def _check_shape(self, output):
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...), "
                             "but given {}".format(y_pred.shape))

        if y_pred.shape[1] != self.num_classes:
            raise ValueError("y_pred does not have correct number of categories: {} vs {}"
                             .format(y_pred.shape[1], self.num_classes))

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...) and y must have "
                             "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                             "but given {} vs {}.".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

        return y_pred, y

    def update(self, output):
        y_pred, y = self._check_shape(output)

        if y_pred.shape != y.shape:
            y_ohe = to_onehot(y.reshape(-1), self.num_classes)
            y_ohe_t = y_ohe.transpose(0, 1).float()
        else:
            y_ohe_t = y.transpose(1, -1).reshape(y.shape[1], -1).float()

        indices = torch.argmax(y_pred, dim=1)
        y_pred_ohe = to_onehot(indices.reshape(-1), self.num_classes)
        y_pred_ohe = y_pred_ohe.float()

        if self.confusion_matrix.type() != y_ohe_t.type():
            self.confusion_matrix = self.confusion_matrix.type_as(y_ohe_t)

        self.confusion_matrix += torch.matmul(y_ohe_t, y_pred_ohe).float()
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        if self.average:
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix.cpu()


class LabelwiseAccuracy(Accuracy):
    def __init__(self, output_transform=lambda x: x):
        self._num_correct = None
        self._num_examples = None
        super(LabelwiseAccuracy, self).__init__(output_transform=output_transform)

    def reset(self):
        self._num_correct = torch.DoubleTensor(0)
        self._num_examples = torch.DoubleTensor(0)
        super(LabelwiseAccuracy, self).reset()

    def update(self, output):

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        num_classes = y_pred.size(1)
        y = to_onehot(y.view(-1), num_classes=num_classes)
        indices = torch.argmax(y_pred, dim=1).view(-1)
        y_pred = to_onehot(indices, num_classes=num_classes)

        y = y.type_as(y_pred)
        correct = y * y_pred
        all_examples = y_pred.sum(dim=0).type(torch.DoubleTensor)

        if correct.sum() == 0:
            true_examples = torch.zeros_like(all_examples)
        else:
            true_examples = correct.sum(dim=0)

        true_examples = true_examples.type(torch.DoubleTensor)

        self._num_correct += true_examples
        self._num_examples += all_examples


    def compute(self):
        if not (isinstance(self._num_examples, torch.Tensor) or self._num_examples > 0):
            raise NotComputableError("{} must have at least one example before"
                                     " it can be computed.".format(self.__class__.__name__))
        return self._num_correct / self._num_examples
