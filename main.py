import argparse
import torch
import copy
import pdb
import gc

from netevaluator import TorchModel


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def memory_dump(core):
    print(
        "Cached memory: {}, Allocated memory: {}".format(
            torch.cuda.memory_cached(device=core),
            torch.cuda.memory_allocated(device=core)))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s → %s:%s%s%s%s %s" %
                        (type(obj).__name__,
                         type(
                            obj.data).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.data.is_pinned else "",
                            " grad" if obj.requires_grad else "",
                            " volatile" if obj.volatile else "",
                            pretty_size(
                            obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)


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
          2: 1e-5,
          3: 1e-6}

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

    print('[MAIN] - Initiating hyperparam evaluation')

    configs = init()

    hyParamChecker = TorchModel()

    for i, config in enumerate(configs):
        print('Creating network for LR [{}] / WIN_SIZE [{}] / WIN_STRIDE [{}]'.format(
            config['lr'], config['win_len'], config['win_step']))

        hyParamChecker.execute_instance(config)
        clean_memory()
        memory_dump(config['gpucore'])
