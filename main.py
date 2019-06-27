import argparse
import torch
import copy
import logging
import gc
import math
import time
import pdb

from windowGenerator import WindowGenerator
from netevaluator import TorchModel

logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
logging.basicConfig(
    filename='debug.log',
    level=logging.INFO,
    format=logging_format)
logger = logging.getLogger('MainLoop')


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
    logger.info(
        "Cached memory: {}, Allocated memory: {}".format(
            convert_size(torch.cuda.memory_cached(device=core)),
            convert_size(torch.cuda.memory_allocated(device=core))))


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

    lr = {0: 1e-3,
          1: 1e-4}

    win_size = {
        0: 70,
        1: 100
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
        'patience': 10,
        'train_info_iter': 10,
        'val_iter': 50,
        'noise': (0, 1e-2),
        'gpucore': 'cuda:0',
        'momentum': 0.9
    }

    if args.type == 1:
        config['channels'] = 39
        config['depth'] = 3
        # config['f_size'] = (3, 5, 1)

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


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", help="Specify GPU core to use")
    parser.add_argument("--type", "-t", help="Specify net type: 0: preMK/postSK, 1: preSK/postMK", default=0, type=int)

    args = parser.parse_args()



    print('[MAIN] - Initiating hyperparam evaluation')
    logger.info('Initiating hyperparam evaluation')

    total_time = time.time()

    configs = init(args)
    
    hyParamChecker = TorchModel(args.type)

    for i, config in enumerate(configs):
        model_time = time.time()
        print('Executing network for LR [{}] / WIN_SIZE [{}] / WIN_STRIDE [{}]'.format(
            config['lr'], config['win_len'], config['win_step']))
        logger.info('Executing network for LR [{}] / WIN_SIZE [{}] / WIN_STRIDE [{}]'.format(
            config['lr'], config['win_len'], config['win_step']))
        memory_dump(config['gpucore'])
        hyParamChecker.execute_instance(config, type=args.type)
        clean_memory()
        memory_dump(config['gpucore'])
        print(' > Took: {:.2} minutes'.format((time.time() - model_time) / 60))
        logger.info('Took: {:.2} minutes'.format((time.time() - model_time) / 60))


    print('FINAL, script took: {:.2} minutes'.format((time.time() - total_time) / 60))
    logger.info('FINAL, script took: {:.2} minutes'.format((time.time() - total_time) / 60))
