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
import csv

from imuCrosstrainer import TorchModel
from tester import Tester

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

    # hyParamChecker = TorchModel(args.freeze, args.lr)
    #
    # pretrained_models = glob.glob(env.pretrained_models_url)
    #
    # # FOR EACH PRETRAINED MODEL
    # for k, model in enumerate(pretrained_models):
    #     model_name = re.search("/([a-zA-Z_]*)\.pth", model).group(1)
    #     # RUN 5 TIMES
    #     for i, iteration in enumerate(range(0,5), start=1):
    #         model_time = time.time()
    #         print("TRAINING PRETRAINED MODEL {}, ITERATION {}".format(model_name, i))
    #         logger.info("[WARN] TRAINING PRETRAINED MODEL {}, ITERATION {}".format(model_name, i))
    #         memory_dump(config['gpucore'])
    #         hyParamChecker.execute_instance(config, model_name, model, i)
    #         print('RESULT [{}]_{} --- Took: {:.6} seconds'.format(i, model_name, time.time() - model_time))
    #         logger.info('[WARN] RESULT [{}]_{} --- Took: {:.6} seconds'.format(i, model_name, time.time() - model_time))
    #
    # print('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    # logger.info('FINAL, script took: {:.6} seconds'.format(time.time() - total_time))
    # print('---------------------------------------------')
    # logger.info('---------------------------------------------')


    types = ['FROZEN', 'UNFROZEN']

    for type in types:
        print(type)
        loaded_models = glob.glob(env.postrained_models_load_url.format(type))
        with open('testResults.csv', mode='w') as csv_file:
            fields = ['name', 'accuracy', 'loss', 'f1', 'acc_[0]', 'acc_[1]', 'acc_[2]', 'acc_[3]', 'acc_[4]', 'acc_[5]', 'acc_[6]']
            for i in range(0, 7):
                for j in range(0, 7):
                    fields.append("CM{}{}".format(i, j))
            fieldsShort = ['name', 'accuracy', 'loss', 'f1']
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            for k, model in enumerate(loaded_models):
                model_name = re.search("-([a-zA-Z]*_[a-zA-Z]*)", model).group(1)
                config["name"] = model_name
                tester = Tester(type=5)
                res = tester.runImuTest(model, config)

                print(res)
                fullr = {**config, **res}
                filtr = {key: value for key, value in fullr.items() if key in fieldsShort}
                accList = res['accPerClass']
                accPc = {"acc_[{}]".format(i): accList[i].item() for i in range(0, len(accList))}
                confMat = res['confMatrix']
                confList = {"CM{}{}".format(i, j): confMat[i][j].item() for i in range(0, len(confMat)) for j in
                            range(0, len(confMat))}
                fullFiltr = {**filtr, **accPc, **confList}
                writer.writerow(fullFiltr)
                clean_memory()
                # memory_dump(args.core)
