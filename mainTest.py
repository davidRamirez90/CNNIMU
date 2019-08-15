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

    types = {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    }

    windows = {
        0: {
            'win_len': 100,
            'win_step': 5
        },
        1: {
            'win_len': 150,
            'win_step': 7
        },
        2: {
            'win_len': 200,
            'win_step': 10
        }
    }

    names = {
        0: "[SK]simple",
        1: "[MK]simple",
        2: "[SK]pretrained",
        3: "[MK]pretrained"
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
            c['name'] = names[i]
            c['win_len'] = windows[j]['win_len']
            c['win_step'] = windows[j]['win_step']
            if types[i] == 1 or types[i] == 3:
                c['channels'] = 38
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
    elif type == 2:
        return "[SK]pretrained"
    elif type == 3:
        return "[MK]pretrained"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", help="Specify GPU core to use", default=0, type=int)
    parser.add_argument("--type", "-t", help="Specify net type: 0: Skeleton, 1: Markers, 2: PreSkeleton, 3: PreMarkers", default=0, type=int)


    args = parser.parse_args()
    # name = getName(args.type)
    configs = init(args)

    with open('testResults.csv', mode='w') as csv_file:
        fields = ['win_len','win_step','name', 'accuracy', 'loss', 'f1', 'accPerClass']
        fieldsNoPerClass = ['win_len','win_step','name', 'accuracy', 'loss', 'f1']
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for i, config in enumerate(configs):
            tester = Tester(type=config['type'])
            name = getName(config['type'])


            for i, iteration in enumerate(range(0,10), start=1):
                res = tester.runTest(config, i)
                print(res)
                fullr = {**config, **res}
                filtr = {key: value for key, value in fullr.items() if key in fieldsNoPerClass}
                pdb.set_trace()
                # writer.writerow(filtr)
                accPC = {key: value for key, value in res['accPerClass']}
                pdb.set_trace()
                fullFiltr = {**filtr, **accPC}
                writer.writerow(fullFiltr)
                clean_memory()
                memory_dump(args.core)
                #pdb.set_trace()
        print('Finished Testing of all models')
        
        
