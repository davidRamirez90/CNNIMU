import argparse
import pdb
import copy
import csv
import gc
import torch
import math

from tester import Tester


def init(args):
    '''
    Initial configuration of used variables
    :return: Array of config objects
    '''


    configArr = []

    # HYPERPARAMETERS
    #     window size
    #     window stride
    #     balancing classes

    lr = {0: 0.0001,
          1: 0.0001,
          2: 0.0001,
          3: 0.0001}

    win_size = {
        0: 100,
        1: 100,
        2: 100,
        3: 100,
    }

    win_stride = {
        0: 5,
        1: 5,
        2: 5,
        3: 5
    }

    # types = {
    #     0: 0,
    #     1: 1,
    #     2: 2,
    #     3: 3,
    #     4: 4
    # }
    types = {
        0: 2
    }

    windows = {
        0: {
            'win_len': 100,
            'win_step': 5
        }
    }

    names = {
        0: "[SK]simple",
        1: "[MK]simple",
        2: "[ACC]simple",
        3: "[SK]pretrained",
        4: "[MK]pretrained"
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
        'momentum': 0.9,
        'win_len': 100,
        'win_step': 5,
        'lr': 0.0001
    }

    if args.core:
        print("Using cuda core: cuda:{}".format(args.core))
        config['gpucore'] = "cuda:{}".format(args.core)

    for i in range(types.__len__()):
        for j in range(windows.__len__()):
            c = copy.deepcopy(config)
            c['type'] = types[i]
            c['name'] = names[types[i]]
            c['win_len'] = windows[j]['win_len']
            c['win_step'] = windows[j]['win_step']
            if types[i] == 1 or types[i] == 2 or types[i] == 4:
                if args.channels == 0:
                    config['channels'] = 38
                else:
                    config['channels'] = args.channels
                c['depth'] = 3
            configArr.append(c)

    # return config
    return configArr

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def memory_dump(core):
    print(
        "Cached memory: {}, Allocated memory: {}".format(
            convert_size(torch.cuda.memory_cached(device=core)),
            convert_size(torch.cuda.memory_allocated(device=core))))

def clean_memory():
    gc.collect()
    # torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def getName(type):
    if type == 0:
        return "[SK]simple"
    elif type == 1:
        return "[MK]simple"
    elif type == 3:
        return "[SK]pretrained"
    elif type == 4:
        return "[MK]pretrained"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", help="Specify GPU core to use", default=0, type=int)
    parser.add_argument("--type", "-t", help="Specify net type: 0: Skeleton, 1: Markers, 2: Deriv ,3: PreSkeleton, 4: PreMarkers", default=0, type=int)
    parser.add_argument("--channels", "-ch", help="Specify number of channels to use", default=0, type=int)


    args = parser.parse_args()
    # name = getName(args.type)
    configs = init(args)
    print(configs)

    with open('testResults.csv', mode='w') as csv_file:
        fields = ['win_len','win_step','name', 'accuracy', 'loss', 'f1', 'acc_[0]', 'acc_[1]', 'acc_[2]', 'acc_[3]', 'acc_[4]', 'acc_[5]', 'acc_[6]']
        for i in range(0, 7):
            for j in range(0, 7):
                fields.append("CM{}{}".format(i, j))
        fieldsShort = ['win_len', 'win_step', 'name', 'accuracy', 'loss', 'f1']
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for i, config in enumerate(configs):
            tester = Tester(type=config['type'])
            name = getName(config['type'])


            for i, iteration in enumerate(range(0,10), start=1):
                res = tester.runTest(config, i)
                print(res)
                fullr = {**config, **res}
                filtr = {key: value for key, value in fullr.items() if key in fieldsShort}
                # writer.writerow(filtr)
                accList = res['accPerClass']
                accPc = { "acc_[{}]".format(i): accList[i].item() for i in range(0, len(accList))}
                confMat = res['confMatrix']
                confList = {"CM{}{}".format(i,j): confMat[i][j].item() for i in range(0, len(confMat)) for j in range(0, len(confMat))}
                #fullFiltr = {**filtr, **accPc}
                fullFiltr = {**filtr, **accPc, **confList}
                writer.writerow(fullFiltr)
                clean_memory()
                memory_dump(args.core)
                #pdb.set_trace()
        print('Finished Testing of all models')
        
        
