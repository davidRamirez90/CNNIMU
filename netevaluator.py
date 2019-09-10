# HELPER IMPORTS
import numpy as np
import logging
import visdom

# TORCH IGNITE IMPORTS
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall, MetricsLambda
from ignite.contrib.handlers import CustomPeriodicEvent, tqdm_logger
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.exceptions import NotComputableError
from ignite.utils import to_onehot


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
# logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
# logging.basicConfig(
#     filename='debug.log',
#     level=logging.INFO,
#     format=logging_format)
# logger = logging.getLogger('Netevaluator')



class TorchModel:
    """
    Allows to evaluate one instance of torch model
    """
    def __init__(self, type, lr, conf):
        print('[netevaluator] - Init Torchmodel')
        self.type = type
        self.lr = lr
        if self.type == 0:
            self.win_url = env.window_url
            self.model_url = env.models_url
            self.envname = "NEWARCH_skeletons".format(conf['win_len'],
                                                      conf['win_step'])
        elif self.type == 1:
            self.win_url = env.marker_window_url
            self.model_url = env.marker_models_url
            self.envname = "NEWARCH_markers".format(conf['win_len'],
                                                    conf['win_step'])
        else:
            self.win_url = env.accel_window_url
            self.model_url = env.accel_models_url
            self.envname = "NEWARCH_accel".format(conf['win_len'],
                                                  conf['win_step'])


    def get_data_loaders(self, config):

        train_batch_size = config['batch_train']
        val_batch_size = config['batch_validate']

        # OBTAINING TRAINING / VALIDATION - DATASET / DATALOADER
        train_dataset = windowDataSet(dir=self.win_url.format(config['win_len'],
                                                          config['win_step'],
                                                          'train'),
                                      transform=GaussianNoise(0, 1e-2, self.type))

        val_dataset = windowDataSet(dir=self.win_url.format(config['win_len'],
                                                        config['win_step'],
                                                        'validate'),
                                    transform=GaussianNoise(0, 1e-2, self.type))

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=True, num_workers=4)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=4)

        return train_loader, val_loader, train_dataset.__len__(), val_dataset.__len__()

    def create_plot_window(self, vis, xlabel, ylabel, title, name=""):
        return vis.line(X=np.array([1]),
                        Y=np.array([np.nan]),
                        name=name,
                        opts=dict(xlabel=xlabel,
                                  ylabel=ylabel,
                                  title=title))

    def append_plot_to_window(self, vis, win, name, update):
        vis.line(X=np.array([1]),
                 Y=np.array([np.nan]),
                 name=name,
                 update=update,
                 win=win)

    def append_scalar_to_plot(self, vis, y, x, update, win, name=""):
        vis.line(Y=[y, ],
                 X=[x, ],
                 name=name,
                 update=update,
                 win=win)

    def F1(self, precision, recall):
        return (precision * recall * 2 / (precision + recall + 1e-20)).mean()

    def score_function(self, engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss


    def execute_instance(self, config, iteration, type=0):

        # CREATING CUSTOM WINDOWS FOR THIS LOOP
        winGen = WindowGenerator(config['win_len'],
                                 config['win_step'],
                                 config['channels'])
        if type == 0:
            win_generated = winGen.run()
        elif type == 1:
            win_generated = winGen.runMarkers()
        else:
            win_generated = winGen.runDerivation()

        assert win_generated


        print('[Main] - Initializing Visdom')
        vis = visdom.Visdom(env="N{}".format(self.envname))

        # GETTING DATA
        train_loader, val_loader, train_size, val_size = self.get_data_loaders(
            config)
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

        # weights = torch.tensor([2.6, 1.5, 1.1, 3.1, 3, 13.8, 6.6])
        # weights = weights.to(device)

        criterion = nn.CrossEntropyLoss()

        # IGNITE METRICS DEFINED INCLUDING CUSTOM F1
        precision = Precision(average=False)
        recall = Recall(average=False)

        metrics = {
            'accuracy': Accuracy(),
            'accPerClass': LabelwiseAccuracy(),
            'loss': Loss(criterion),
            'precision': precision,
            'recall': recall,
            'f1': MetricsLambda(self.F1, precision, recall)
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
            dirname=self.model_url,
            filename_prefix='[{}]-CNNIMU_{}_{}_{}'.format(
                iteration,
                config['win_len'],
                config['win_step'],
                config['lr']),
            score_function=self.score_function,
            score_name='loss',
            create_dir=True,
            require_empty=False)
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                        checkpoint,
                                        {'network': net})

        earlyStopper = EarlyStopping(patience=config['patience'],
                                     score_function=self.score_function,
                                     trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, earlyStopper)

        # LR ADAPTER HANDLER
        step_scheduler = ReduceLROnPlateau(
            optimizer,
            mode = 'min',
            factor = 0.1,
            patience = 5,
            verbose = True)



        # CREATING VISDOM INITIAL GRAPH OBJECTS

        train_metrics_window = self.create_plot_window(
            vis,
            '# Iterations',
            'Loss',
            '[{}] Val / Train Losses W [{}/{}] - LR [{}]'.format(
                iteration,
                config['win_len'],
                config['win_step'],
                config['lr']),
            'trainingloss')
        self.append_plot_to_window(
            vis,
            train_metrics_window,
            'validationloss',
            'append')
        val_acc_window = self.create_plot_window(
            vis,
            '# Iterations',
            'Accuracy',
            '[{}] Validation Accuracy W [{}/{}] - LR [{}]'.format(
                iteration,
                config['win_len'],
                config['win_step'],
                config['lr']))
        # ACCURACY PER CLASS VISDOM CONFIG
        acc_per_class_window = self.create_plot_window(
            vis,
            '# Iterations',
            'Accuracy',
            '[{}] Acc per class W [{}/{}] - LR [{}]'.format(
                iteration,
                config['win_len'],
                config['win_step'],
                config['lr']),
            '0')
        self.append_plot_to_window(vis, acc_per_class_window, '1', 'append')
        self.append_plot_to_window(vis, acc_per_class_window, '2', 'append')
        self.append_plot_to_window(vis, acc_per_class_window, '3', 'append')
        self.append_plot_to_window(vis, acc_per_class_window, '4', 'append')
        self.append_plot_to_window(vis, acc_per_class_window, '5', 'append')
        self.append_plot_to_window(vis, acc_per_class_window, '6', 'append')

        # val_f1_window = self.create_plot_window(
        #     vis,
        #     '# Iterations',
        #     'F1',
        #     '[{}] F1 score W [{}/{}] - LR [{}]'.format(
        #         iteration,
        #         config['win_len'],
        #         config['win_step'],
        #         config['lr']))

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

        @trainer.on(val_cpe.Events.ITERATIONS_90_COMPLETED)
        def run_validation(engine):
            val_evaluator.run(val_loader)

        @val_evaluator.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            m = engine.state.metrics
            if self.lr:
                step_scheduler.step(m['loss'])
            self.append_scalar_to_plot(vis, m['loss'],
                                       trainer.state.iteration,
                                       'append', train_metrics_window,
                                       name='validationloss')
            self.append_scalar_to_plot(vis, m['accuracy'],
                                       trainer.state.iteration,
                                       'append', val_acc_window)
            self.append_scalar_to_plot(vis, m['accPerClass'][0], trainer.state.iteration, 'append',
                                       acc_per_class_window, name='0')
            self.append_scalar_to_plot(vis, m['accPerClass'][1], trainer.state.iteration, 'append',
                                       acc_per_class_window, name='1')
            self.append_scalar_to_plot(vis, m['accPerClass'][2], trainer.state.iteration, 'append',
                                       acc_per_class_window, name='2')
            self.append_scalar_to_plot(vis, m['accPerClass'][3], trainer.state.iteration, 'append',
                                       acc_per_class_window, name='3')
            self.append_scalar_to_plot(vis, m['accPerClass'][4], trainer.state.iteration, 'append',
                                       acc_per_class_window, name='4')
            self.append_scalar_to_plot(vis, m['accPerClass'][5], trainer.state.iteration, 'append',
                                       acc_per_class_window, name='5')
            self.append_scalar_to_plot(vis, m['accPerClass'][6], trainer.state.iteration, 'append',
                                       acc_per_class_window, name='6')
            # self.append_scalar_to_plot(vis, m['f1'],
            #                            trainer.state.iteration,
            #                            'append', val_f1_window)
            print(
                "Validation Result: ----------->  Loss: {:.4f}, Accuracy: {:.4f}, F1: {:.4f}".format(
                    m['loss'],
                    m['accuracy'],
                    m['f1']))


        trainer.run(train_loader, max_epochs=15)

        # logger.info('Finished training after {} iterations'.format(trainer.state.iteration))
        del training_losses_acc
        del trainer
        del val_evaluator
        del step_scheduler
        del net




class GaussianNoise(object):
    """
    Add Gaussian noise to a window data sample
    """
    
    def __init__(self, mu, sigma, type):
        self.mu = mu
        self.sigma = sigma
        self.type = type

    def __call__(self, sample):
        data = sample['data']
        label = np.long(sample['label'])
        data += np.random.normal(self.mu,
                                 self.sigma,
                                 data.shape)
        # THIS TWO TYPES BELONG TO SKELETON TRAINING NEED EXPANDED DIMS
        if self.type == 0 or self.type == 4:
            data = np.expand_dims(data, 0)
        return (data, label)


class LabelwiseAccuracy(Accuracy):
    def __init__(self, output_transform=lambda x: x):
        self._num_correct = None
        self._num_examples = None
        super(LabelwiseAccuracy, self).__init__(output_transform=output_transform)

    def reset(self):
        self._num_correct = torch.DoubleTensor(0)
        self._num_examples = torch.DoubleTensor(0)
        super(LabelwiseAccuracy, self).reset()

    def update(self, output):

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        num_classes = y_pred.size(1)
        y = to_onehot(y.view(-1), num_classes=num_classes)
        indices = torch.argmax(y_pred, dim=1).view(-1)
        y_pred = to_onehot(indices, num_classes=num_classes)

        y = y.type_as(y_pred)
        correct = y * y_pred
        all_examples = y_pred.sum(dim=0).type(torch.DoubleTensor)

        if correct.sum() == 0:
            true_examples = torch.zeros_like(all_examples)
        else:
            true_examples = correct.sum(dim=0)

        true_examples = true_examples.type(torch.DoubleTensor)

        self._num_correct += true_examples
        self._num_examples += all_examples

    def compute(self):
        if not (isinstance(self._num_examples, torch.Tensor) or self._num_examples > 0):
            raise NotComputableError("{} must have at least one example before"
                                     " it can be computed.".format(self.__class__.__name__))
        return self._num_correct / self._num_examples
