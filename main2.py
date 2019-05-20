
# HELPER IMPORTS
import numpy as np
import argparse
import logging
import visdom
import copy

# TORCH IGNITE IMPORTS
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall, MetricsLambda
from ignite.contrib.handlers import CustomPeriodicEvent, tqdm_logger
from ignite.handlers import EarlyStopping, ModelCheckpoint

# TORCH IMPORTS
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import nn
import torch

# CUSTOM SCRIPTS IMPORT SECTION
from windowGenerator import WindowGenerator
from windataset import windowDataSet
from network import CNN_IMU
import env
import pdb


# INITIAL CONFIG OF VARIABLES
url = env.window_url
logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
logging.basicConfig(
    filename='debug3.log',
    level=logging.DEBUG,
    format=logging_format)
logger = logging.getLogger('CNN network')


class GaussianNoise(object):
    """
    Add Gaussian noise to a window data sample
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        data = sample['data']
        label = np.long(sample['label'])
        data += np.random.normal(self.mu,
                                 self.sigma,
                                 data.shape)
        data = np.expand_dims(data, 0)
        return (data, label)


def get_data_loaders(config):

    train_batch_size = config['batch_train']
    val_batch_size = config['batch_validate']

    # OBTAINING TRAINING / VALIDATION - DATASET / DATALOADER
    train_dataset = windowDataSet(dir=url.format(config['win_len'],
                                                 config['win_step'],
                                                 'train'),
                                  transform=GaussianNoise(0, 1e-2))

    val_dataset = windowDataSet(dir=url.format(config['win_len'],
                                               config['win_step'],
                                               'validate'),
                                transform=GaussianNoise(0, 1e-2))

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, val_loader, train_dataset.__len__(), val_dataset.__len__()


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
          3: 1e-5,
          4: 1e-6}

    win_size = {
        0: 70,
        1: 85,
        2: 100
    }

    win_stride = {
        0: 5,
        1: 1
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
        logger.info("Selected cuda core: cuda:{}".format(args.core))
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


def F1(precision, recall):
    return (precision * recall * 2 / (precision + recall + 1e-20)).mean()


def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss


def create_plot_window(vis, xlabel, ylabel, title, name=""):
    return vis.line(X=np.array([1]),
                    Y=np.array([np.nan]),
                    name=name,
                    opts=dict(xlabel=xlabel,
                              ylabel=ylabel,
                              title=title))


def append_plot_to_window(vis, win, name, update):
    vis.line(X=np.array([1]),
             Y=np.array([np.nan]),
             name=name,
             update=update,
             win=win)


def append_scalar_to_plot(vis, y, x, update, win, name=""):
    vis.line(Y=[y, ],
             X=[x, ],
             name=name,
             update=update,
             win=win)


def run(i, config):
    """
    :param i: Current hyperparam loop iteration
    :param config: Configuration object
    :return: None
    """

    # CREATING CUSTOM WINDOWS FOR THIS LOOP
    winGen = WindowGenerator(config['win_len'],
                             config['win_step'])
    win_generated = winGen.run()
    assert win_generated

    print('[Main] - Initializing Visdom')
    vis = visdom.Visdom(env='IGNITE_workspace')

    # GETTING DATA
    train_loader, val_loader, train_size, val_size = get_data_loaders(config)

    # NETWORK CREATION
    device = torch.device(
        config['gpucore'] if torch.cuda.is_available() else "cpu")
    net = CNN_IMU(config)
    net = net.to(device)

    print(device)
    print(net)

    # OPTIMIZER AND CRITERION INITIALIZATION
    optimizer = SGD(net.parameters(),
                    lr=config['lr'],
                    momentum=config['momentum'])
    criterion = nn.CrossEntropyLoss()

    # IGNITE METRICS DEFINED INCLUDING CUSTOM F1
    precision = Precision(average=False)
    recall = Recall(average=False)

    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion),
        'precision': precision,
        'recall': recall,
        'f1': MetricsLambda(F1, precision, recall)
    }

    # IGNITE TRAINER AND EVAL OBJECTS CONFIG
    trainer = create_supervised_trainer(net,
                                        optimizer,
                                        criterion,
                                        device=device)
    val_evaluator = create_supervised_evaluator(
        net, metrics=metrics, device=device)

    # LIFETIME EVENTS FOR PRINTING CALCULATING AND PLOTTING
    tr_cpe = CustomPeriodicEvent(n_iterations=config['train_info_iter'])
    val_cpe = CustomPeriodicEvent(n_iterations=config['val_iter'])
    tr_cpe.attach(trainer)
    val_cpe.attach(trainer)

    # TQDM OBSERVERS
    pbar = tqdm_logger.ProgressBar()
    pbar.attach(val_evaluator)

    # CREATING EARLY STOPPING AND SAVE HANDLERS
    checkpoint = ModelCheckpoint(
        dirname='/data/dramirez/models',
        filename_prefix='CNNIMU_{}_{}_{}'.format(
            config['win_len'],
            config['win_step'],
            config['lr']),
        score_function=score_function,
        score_name='loss',
        create_dir=True,
        require_empty=False)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                    checkpoint,
                                    {'network': net})

    earlyStopper = EarlyStopping(patience=config['patience'],
                                 score_function=score_function,
                                 trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, earlyStopper)

    # CREATING VISDOM INITIAL GRAPH OBJECTS

    train_metrics_window = create_plot_window(
        vis, '# Iterations', 'Loss', 'Val / Train Losses W [{}/{}] - LR [{}]'.format(
            config['win_len'], config['win_step'], config['lr']), 'trainingloss')
    append_plot_to_window(
        vis,
        train_metrics_window,
        'validationloss',
        'append')
    val_acc_window = create_plot_window(
        vis,
        '# Iterations',
        'Accuracy',
        'Validation Accuracy W [{}/{}] - LR [{}]'.format(
            config['win_len'],
            config['win_step'],
            config['lr']))
    val_f1_window = create_plot_window(
        vis, '# Iterations', 'F1', 'F1 score W [{}/{}] - LR [{}]'.format(
            config['win_len'], config['win_step'], config['lr']))

    training_losses_acc = list()

    # IGNITE EVENTS DEFINITION
    @trainer.on(Events.EPOCH_STARTED)
    def initial_eval(engine):
        val_evaluator.run(val_loader)

    @trainer.on(Events.ITERATION_COMPLETED)
    def accumulate_trainlosses(engine):
        training_losses_acc.append(engine.state.output)

    @trainer.on(tr_cpe.Events.ITERATIONS_10_COMPLETED)
    def log_training_loss(engine):
        # breakpoint()
        vis.line(Y=np.array(training_losses_acc),
                 X=np.arange(start=0, stop=training_losses_acc.__len__()),
                 name='trainingloss',
                 update='replace',
                 win=train_metrics_window)
        print(
            "Epoch[{}],  Iteration[{}],  Loss: {:.2f}".format(
                engine.state.epoch,
                engine.state.iteration,
                engine.state.output))

    @trainer.on(val_cpe.Events.ITERATIONS_50_COMPLETED)
    def run_validation(engine):
        val_evaluator.run(val_loader)

    @val_evaluator.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        m = engine.state.metrics
        append_scalar_to_plot(vis, m['loss'],
                              trainer.state.iteration,
                              'append', train_metrics_window,
                              name='validationloss')
        append_scalar_to_plot(vis, m['accuracy'],
                              trainer.state.iteration,
                              'append', val_acc_window)
        append_scalar_to_plot(vis, m['f1'],
                              trainer.state.iteration,
                              'append', val_f1_window)
        print("Validation Result: ----------------->  Loss: {:.4f}, Accuracy: {:.4f}, F1: {:.4f}".format(
            trainer.state.epoch, trainer.state.iteration, m['loss'], m['accuracy'], m['f1']))

    trainer.run(train_loader, max_epochs=2)
    pdb.set_trace()


if __name__ == '__main__':

    configs = init()

    for i, config in enumerate(configs):
        print('Creating network for LR [{}] / WIN_SIZE [{}] / WIN_STRIDE [{}]'.format(
            config['lr'], config['win_len'], config['win_step']))
        run(i, config)
