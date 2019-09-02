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
        0: 100,
        1: 70
    }

    win_stride = {
        0: 5
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
        'val_iter': 90,
        'noise': (0, 1e-2),
        'gpucore': 'cuda:0',
        'momentum': 0.9,
        'win_len': 100,
        'win_step': 5,
        'lr': 0.0001
    }

    if args.type == 1 or args.type == 2:
        if args.channels != 0:
            config['channels'] = 38
        else:
            config['channels'] = args.channels
        config['depth'] = 3



    if args.core:
        print("Using cuda core: cuda:{}".format(args.core))
        config['gpucore'] = "cuda:{}".format(args.core)

    # for i in range(win_size.__len__()):
    #     for j in range(win_stride.__len__()):
    #         for k in range(lr.__len__()):
    #             c = copy.deepcopy(config)
    #             c['win_len'] = win_size[i]
    #             c['win_step'] = win_stride[j]
    #             c['lr'] = lr[k]
    #             configArr.append(c)

    return config


def clean_memory():
    gc.collect()
    # torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", help="Specify GPU core to use")
    parser.add_argument("--channels", "-ch", help="Specify number of channels to use", default=0, type=int)
    parser.add_argument("--type", "-t", help="Specify net type: 0: Skeletons, 1: Markers, 2: Deriv", default=0, type=int)
    parser.add_argument("--lr", "-l", help="Specify if LR reduction used", default=False, type=bool)


    args = parser.parse_args()



    print('[MAIN] - Initiating hyperparam evaluation')
    logger.info('Initiating hyperparam evaluation')

    total_time = time.time()

    configs = init(args)
    print(configs)
    hyParamChecker = TorchModel(args.type, args.lr, configs)

    for i, iteration in enumerate(range(0,10), start=1):
        model_time = time.time()
        print('Executing TRAINING for MODE [{}] / ITERATION [{}]'.format(
            args.type, i))
        logger.info('Executing training for MODE [{}] / ITERATION [{}]'.format(
            args.type, i))
        # memory_dump(configs['gpucore'])
        hyParamChecker.execute_instance(configs, i, type=args.type)
        clean_memory()
        # memory_dump(configs['gpucore'])
        print('TRAINING ---[# {}] --- [m: {}] --- Took: {:.6} seconds'.format(i, args.type, time.time() - model_time))
        logger.info('TRAINING ---[# {}] --- [m: {}] --- Took: {:.6} seconds'.format(i, args.type, time.time() - model_time))

    # for i, config in enumerate(configs):
    #     model_time = time.time()
    #     print('Executing network for LR [{}] / WIN_SIZE [{}] / WIN_STRIDE [{}]'.format(
    #         config['lr'], config['win_len'], config['win_step']))
    #     logger.info('Executing network for LR [{}] / WIN_SIZE [{}] / WIN_STRIDE [{}]'.format(
    #         config['lr'], config['win_len'], config['win_step']))
    #     memory_dump(config['gpucore'])
    #     hyParamChecker.execute_instance(config, type=args.type)
    #     clean_memory()
    #     memory_dump(config['gpucore'])
    #     print(' > Took: {:.2} seconds'.format(time.time() - model_time))
    #     logger.info('Took: {:.2} seconds'.format(time.time() - model_time))


    print('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    logger.info('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    print('---------------------------------------------')
    logger.info('---------------------------------------------')
