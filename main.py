from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import visdom
import numpy as np
import tqdm
import env
import logging
from torch.autograd import Variable
from network import CNN_IMU
from windataset import windowDataSet
import copy
import argparse
from earlyStopping import earlyStopper

# INITIAL CONFIG OF VARIABLES
url = env.window_url
logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
logging.basicConfig(filename='debug3.log', level=logging.DEBUG, format=logging_format)
logger = logging.getLogger('CNN network')



def addGaussianNoise(data, mu, sigma):
    '''
    :param data: Data tensor
    :param mu: Mean of Noise Matrix
    :param sigma: Std dev
    :return: res: Data + noise
    '''
    noise = Variable(data.new(data.size()).normal_(mu, sigma))
    res = data.add_(noise)
    return res


def get_accuracy(pred, targets):
    """
    :param pred: Predictions tensor
    :param targets: Targets tensor
    :return: returns: Acc, correct, total
    """
    pred = F.softmax(pred, 1)
    pred = pred.argmax(dim=1)
    correct = torch.sum(torch.eq(targets, pred)).item()
    total = targets.size()[0]
    acc = (correct/total)*100
    return acc, correct, total

def init():
    '''
    Initial configuration of used variables
    :return: Array of config objects
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", help="Specify GPU core to use")
    args = parser.parse_args()

    configArr = []

    # HYPERPARAMETERS
    #     window size
    #     window stride
    #     balancing classes

    lr = {0: 1e-2,
          1: 1e-3,
          2: 1e-4,
          3: 1e-5}

    config = {
        'in_dim': (100,132),
        'n_classes': 7,
        'n_filters': 64,
        'f_size': (5,1),
        'batch_train': 100,
        'batch_validate': 100,
        'patience': 7,
        'train_info_iter': 10,
        'validate_iter': 40,
        'win_len': 100,
        'win_step': 1,
        'noise': (0, 1e-2),
        'gpucore': 'cuda:0'
    }

    if (args.core):
        print("Using cuda core: cuda:{}".format(args.core))
        logger.info("Selected cuda core: cuda:{}".format(args.core))
        config['gpucore'] = "cuda:{}".format(args.core)

    for i in range(3):
        c = copy.deepcopy(config)
        c['lr'] = lr[i]
        configArr.append(c)

    return configArr


def createNetwork(i, config):
    print(config)
    earlyStop = earlyStopper(7, config)

    # USE GPU IF AVAILABLE ELSE CPU
    device = torch.device(config['gpucore'] if torch.cuda.is_available() else "cpu")

    # OBTAINING TRAINING / VALIDATION - DATASET / DATALOADER
    train_dataset = windowDataSet(dir=url.format('train'))
    validate_dataset = windowDataSet(dir=url.format('validate'))
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_train'], shuffle=True, num_workers=4)
    validate_dataloader = DataLoader(validate_dataset, batch_size=config['batch_validate'], shuffle=False, num_workers=4)

    # NETWORK CREATION
    net = CNN_IMU(config)
    net = net.to(device)

    # OPTIMIZER AND CRITERION INITIALIZATION
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    vis = visdom.Visdom()

    trWin = vis.line(np.zeros(1), np.zeros(1), name='trainloss', opts=dict(
        title='Training network: {}'.format(i),
        caption='random caption',
        ytickmin=0,
        ytickmax=3, ))
    vis.line(np.zeros(1), np.zeros(1), name='validationloss', update='append', win=trWin)

    train_losses = list()
    validate_losses = list()
    validate_epochs = list()
    val_epochs_base = 0
    best_val = float("inf")

    breakTraining = False

    for epoch in range(4):
        procdata = 0
        corrP = 0
        totP = 0
        print('epoch {}'.format(epoch))
        logger.info('epoch {}'.format(epoch))
        for i_batch, batch_data in enumerate(train_dataloader):
            net.train(True)
            x = batch_data['data'].unsqueeze(dim=1)
            x = x.to(device)
            x = addGaussianNoise(x, 0, 1e-2)
            target = batch_data['label'].long()
            target = target.to(device)
            # NETWORK PREDICTION
            output = net(x)
            acc, c, t = get_accuracy(output, target)
            corrP += c
            totP += t
            # ZERO GRADIENTS AND PERFORM STEP
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # KEEP TRACK OF TRAINING LOSSES, PROCESSED DATA AND RUNNING ACC
            train_losses.append(loss.item())
            procdata += train_dataloader.batch_size
            eAcc = (corrP / totP) * 100


            # UPDATE TRAIN FUNCTION LOSS
            if i_batch % 9 == 0:
                print(' Epoch: {} - Iterations: {}  Processed {} / {}'.format(epoch, i_batch, procdata,
                                                                              train_dataset.__len__()))
                print('     Loss value: {:06}   Accuracy: {:02}%   -- Avg:  {:04}'.format(loss.item(), acc, eAcc))
                logger.info(' Epoch: {} - Iterations: {}  Processed {} / {}'.format(epoch, i_batch, procdata,
                                                                                    train_dataset.__len__()))
                logger.info('     Loss value: {:06}   Accuracy: {:02}%   -- Avg:  {:04}'.format(loss.item(), acc, eAcc))
                vis.line(Y=np.array(train_losses),
                         X=np.arange(start=0, stop=train_losses.__len__()),
                         name='trainloss',
                         update='replace',
                         win=trWin, )
            # UPDATE VALIDATION FUNCTION LOSS
            if i_batch % 30 == 0:
                print('STARTING VALIDATION PHASE')
                logger.info('STARTING VALIDATION PHASE')
                net.train(False)
                net.eval()
                corrP_v = 0
                totP_v = 0
                totLoss = 0
                procdata_v = 0
                dslen = validate_dataset.__len__()
                with torch.no_grad():
                    for i_batchv, v_batch_data in enumerate(tqdm.tqdm(validate_dataloader)):
                        x_v = v_batch_data['data'].unsqueeze(dim=1)
                        x_v = x_v.to(device)
                        target_v = v_batch_data['label'].long()
                        target_v = target_v.to(device)
                        output_v = net(x_v)
                        acc_v, c_v, t_v = get_accuracy(output_v, target_v)
                        corrP_v += c_v
                        totP_v += t_v
                        loss_v = criterion(output_v, target_v)
                        optimizer.zero_grad()
                        procdata_v += validate_dataloader.batch_size
                        # print('Processes - {} / {}'.format(procdata_v, validate_dataset.__len__()))
                        totLoss += (loss_v * x_v.shape[0] / dslen)
                    cpuloss = totLoss.cpu().numpy()
                    validate_losses.append(cpuloss)
                    validate_epochs.append(val_epochs_base)
                    print('Best {} / Curr {}'.format(best_val, cpuloss))
                    if cpuloss < best_val:
                        best_val = cpuloss
                        vis.line([0, 3],
                                 [val_epochs_base, val_epochs_base],
                                 name='bestTillNow',
                                 update='replace',
                                 win=trWin)
                    val_epochs_base += 30
                    print('VALIDATION   epoch {},  Iteration {}'.format(epoch, i_batch))
                    print('RESULSTS     Loss Value: {:06}'.format(totLoss))
                    logger.info('VALIDATION   epoch {},  Iteration {}'.format(epoch, i_batch))
                    logger.info('RESULSTS     Loss Value: {:06}'.format(totLoss))
                    vis.line(Y=np.array(validate_losses),
                             X=validate_epochs,
                             name='validationloss',
                             update='replace',
                             win=trWin, )
                    breakTraining = earlyStop.step(cpuloss, loss.item())
                    if breakTraining:
                        break

        print('----------------------')
        print('Epoch Accuracy:  {}%'.format(eAcc))
        print('----------------------')
        logger.info('----------------------')
        logger.info('Epoch Accuracy:  {}%'.format(eAcc))
        logger.info('----------------------')
        if breakTraining:
            break


if __name__ == '__main__':

    configs = init()

    for i, config in enumerate(configs):
        createNetwork(i, config)












