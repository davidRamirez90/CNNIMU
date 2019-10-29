import argparse
import torch
import copy
import logging
import gc
import math
import time
import pdb
import env
import glob
import re

from imuCrosstrainer import TorchModel

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
    assert (isinstance(size, torch.Size))
    return " ? ".join(map(str, size))


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
    config = {
        'channels': 30,
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
        'momentum': 0.9,
        'win_len': 50,
        'win_step': 2,
        'lr': 0.0001
    }

    if args.core:
        print("Using cuda core: cuda:{}".format(args.core))
        config['gpucore'] = "cuda:{}".format(args.core)

    return config


def clean_memory():
    gc.collect()
    #  torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--core", "-c", help="Specify GPU core to use")
    parser.add_argument("--freeze", "-f", help="Specify if Conv Layers should be frozen", default=False, type=bool)
    parser.add_argument("--lr", "-l", help="Specify if LR reduction used", default=True, type=bool)

    args = parser.parse_args()

    print('[MAIN] - Initiating hyperparam evaluation')
    logger.info('Initiating hyperparam evaluation')

    total_time = time.time()

    config = init(args)

    print(config)

    hyParamChecker = TorchModel(args.freeze, args.lr)

    pretrained_models = glob.glob(env.pretrained_models_url)

    # FOR EACH PRETRAINED MODEL
    for k, model in enumerate(pretrained_models):
        model_name = re.search("/([a-zA-Z_]*)\.pth", model).group(1)
        # RUN 5 TIMES
        for i, iteration in enumerate(range(0,5), start=1):
            model_time = time.time()
            print("TRAINING PRETRAINED MODEL {}, ITERATION {}".format(model_name, i))
            logger.info("[WARN] TRAINING PRETRAINED MODEL {}, ITERATION {}".format(model_name, i))
            memory_dump(config['gpucore'])
            hyParamChecker.execute_instance(config, model_name, model, i)
            print('RESULT [{}]_{} --- Took: {:.6} seconds'.format(i, model_name, time.time() - model_time))
            logger.info('[WARN] RESULT [{}]_{} --- Took: {:.6} seconds'.format(i, model_name, time.time() - model_time))

    print('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    logger.info('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    print('---------------------------------------------')
    logger.info('---------------------------------------------')



    # for i, iteration in enumerate(range(0, 10), start=1):
    #     model_time = time.time()
    #     print('Executing CROSSTRAINING for MODE [{}] / ITERATION [{}]'.format(
    #         args.type, i))
    #     logger.info('Executing training for MODE [{}] / ITERATION [{}]'.format(
    #         args.type, i))
    #     memory_dump(configs['gpucore'])
    #     hyParamChecker.execute_instance(configs, i, type=args.type)
    #     # memory_dump(configs['gpucore'])
    #     clean_memory()
    #     memory_dump(configs['gpucore'])
    #     print('CROSSTRAINING ---[# {}] --- [m: {}] --- Took: {:.6} seconds'.format(i, args.type,
    #                                                                                time.time() - model_time))
    #     logger.info('CROSSTRAINING ---[# {}] --- [m: {}] --- Took: {:.6} seconds'.format(i, args.type,
    #                                                                                      time.time() - model_time))

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
    #     print(' > Took: {:.2} minutes'.format((time.time() - model_time) / 60))
    #     logger.info('Took: {:.2} minutes'.format((time.time() - model_time) / 60))

    print('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    logger.info('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    print('---------------------------------------------')
    logger.info('---------------------------------------------')
