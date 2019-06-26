import argparse
import pdb
import copy
import csv

from tester import Tester


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

    lr = {0: 1e-3,
          1: 1e-4,
          2: 1e-5}

    win_size = {
        0: 70,
        1: 85,
        2: 100
    }

    win_stride = {
        0: 1,
        1: 5
    }

    config = {
        'channels': 132,
        'depth': 1,
        'n_classes': 7,
        'n_filters': 64,
        'f_size': (5, 1),
        'batch_train': 100,
        'batch_validate': 100,
        'patience': 7,
        'train_info_iter': 10,
        'val_iter': 50,
        'noise': (0, 1e-2),
        'gpucore': 'cuda:0',
        'momentum': 0.9
    }

    if args.type == 1:
        config['channels'] = 39
        config['depth'] = 3

    if args.core:
        print("Using cuda core: cuda:{}".format(args.core))
        config['gpucore'] = "cuda:{}".format(args.core)

    for i in range(win_size.__len__()):
        for j in range(win_stride.__len__()):
            for k in range(lr.__len__()):
                c = copy.deepcopy(config)
                c['win_len'] = win_size[i]
                c['win_step'] = win_stride[j]
                c['lr'] = lr[k]
                configArr.append(c)

    return configArr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", help="Specify GPU core to use", default=0, type=int)
    parser.add_argument("--type", "-t", help="Specify net type: 0: Skeleton, 1: Markers", default=0, type=int)
    parser.add_argument("--name", "-n", help="Specify file name to save", default="testResults")

    args = parser.parse_args()
    
    configs = init()
    tester = Tester(type=args.type)
    with open('{}.csv'.format(args.name), mode='w') as csv_file:
        fields = ['win_len', 'win_step', 'lr', 'accuracy', 'loss', 'f1']
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for i, config in enumerate(configs):
            res = tester.runTest(config)
            fullr = {**config, **res}
            filtr = {key: value for key, value in fullr.items() if key in fields}
            writer.writerow(filtr)
            #pdb.set_trace()
        print('Finished Testing of all models')
        
        
