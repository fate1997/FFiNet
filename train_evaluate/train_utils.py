from torch_geometric.loader import DataLoader
from typing import Tuple
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem


class TargetSelectTransform(object):
    def __init__(self, target_id=0):
        self.target_id = target_id

    def __call__(self, data):
        data.y = data.y[:, :12]
        return data


def data2iter(data_path: str,
              train_args, 
              seed,
              split_ratio: Tuple = (0.8, 0.1, 0.1)):

    # load dataset and split to train and test set
    dataset = torch.load(data_path)
    split_1 = int(split_ratio[0] * len(dataset))
    split_2 = int((split_ratio[0]+split_ratio[1]) * len(dataset))

    r = random.random
    random.seed(seed)
    random.shuffle(dataset, random=r)
    if train_args.task_name[0] == 'pdbbind':
        split_1 = int(0.9 * len(dataset))
        train_loader = DataLoader(dataset[:split_1], batch_size=train_args.batch_size)
        val_loader = DataLoader(dataset[split_1:], batch_size=train_args.batch_size)
        dataset_test_path = data_path.rsplit('\\', 1)[0] + '\\pdbbind_core_' + data_path.split('\\')[-1].split('_')[-1]
        dataset_test = torch.load(dataset_test_path)
        test_loader = DataLoader(dataset_test, batch_size=train_args.batch_size)
    else:
        train_loader = DataLoader(dataset[:split_1], batch_size=train_args.batch_size)
        val_loader = DataLoader(dataset[split_1:split_2], batch_size=train_args.batch_size)
        test_loader = DataLoader(dataset[split_2:], batch_size=train_args.batch_size)

    if train_args.normalize == True:
        labels = []
        for data in train_loader:
            labels.append(data.y.reshape(-1, train_args.num_tasks))
        labels = torch.cat(labels)
        y_mean = labels.mean(0).reshape((-1, train_args.num_tasks))
        y_std = labels.std(0).reshape((-1, train_args.num_tasks))
        y_mad = []
        for i in range(train_args.num_tasks):
            mad = pd.Series(labels[:, i]).mad()
            y_mad.append(mad)
        y_mad = torch.FloatTensor(y_mad)
        train_args.y_ratio = (y_std/y_mad).squeeze(0).to(train_args.device)
        train_args.y_mean = y_mean.to(train_args.device)
        train_args.y_std = y_std.to(train_args.device)

    return train_loader, val_loader, test_loader


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(self, train_args):
        self.trainargs = train_args
        self.patience = train_args.patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.tolerance = train_args.tolerance
        self.path = train_args.model_save_path
        self.model_type = train_args.model_type

    def __call__(self, val_loss, test_loader, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, test_loader, model)
        elif score < self.best_score * (1-self.tolerance):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, test_loader, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, test_loader, model):
        # Saves model when validation score doesn't improve in patience
        torch.save(model.state_dict(), self.path + self.model_type + '.pt')
        save_results(model, test_loader, self.trainargs)
        self.val_loss_min = val_loss


# flatten batch
def batch_flatten(model, data_loader, train_args=None):
    model.eval()
    device = train_args.device
    model.to(device)
    y_true = []
    y_predict = []
    smiles = []
    device, normalize, num_tasks = train_args.device, train_args.normalize, train_args.num_tasks
    for batch in data_loader:
        batch = batch.to(device)
        if normalize:
            y_hat = (model(batch).detach().reshape((-1, num_tasks)) * train_args.y_std + train_args.y_mean).tolist()
            y_true += (batch.y.reshape((-1, num_tasks))).tolist()
        else:
            y_hat = model(batch).detach().reshape((-1, num_tasks)).tolist()
            y_true += batch.y.reshape((-1, num_tasks)).tolist()
        y_predict += y_hat
        smiles += batch.smiles
    return y_true, y_predict, smiles


# preview results
def plotting(x, y, xlabel, ylabel):
    plt.figure(figsize=(8, 8))

    if type(x) == torch.Tensor:
        x, y = x.cpu().detach(), y.cpu().detach()
    x_min, y_min = min(x), min(y)
    x_min, y_min = min(x_min, y_min), min(x_min, y_min)
    x_max, y_max = max(x) * 1.2, max(y) * 1.2
    x_max, y_max = max(x_max, y_max), max(x_max, y_max)
    plt.plot([x_min, x_max], [y_min, y_max])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.scatter(x, y, marker='x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def evaluate_score(model, data_loader, train_args):
    if isinstance(model, nn.Module):
        model.eval()
    
    y_true, y_pred, _ = batch_flatten(model, data_loader,
                                    train_args=train_args)

    metric_dict = {}
    if train_args.task == 'regression':
        metric_dict['RMSE'] = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        metric_dict['multi-MAE'] = (np.abs((np.array(y_true) - np.array(y_pred))).mean(0)).tolist()
        metric_dict['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
        metric_dict['loss'] = nn.MSELoss()(torch.tensor(y_true, dtype=torch.float32), 
                                        torch.tensor(y_pred, dtype=torch.float32))
    elif train_args.task == 'binary':
        y_pred = torch.sigmoid(torch.tensor(y_pred)).tolist()
        metric_dict['ROC-AUC'] = metrics.roc_auc_score(y_true, y_pred)
        metric_dict['Accuracy'] = metrics.average_precision_score(y_true, y_pred)
        metric_dict['loss'] = nn.BCEWithLogitsLoss()(torch.tensor(y_true, dtype=torch.float32), 
                                        torch.tensor(y_pred, dtype=torch.float32))
    return metric_dict

class TrainArgs:
    def __init__(self, num_epochs = 10000, lr = 0.001, batch_size = 128, model_save_path = '.\\train_evaluate\\saved_models\\', 
                model_type = None, patience = 100, task = 'regression', num_tasks = 1, normalize = False, 
                interval = 10, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                metrics = 'RMSE', task_name = ['ESOL'], tolerance=0, split='random', results_dir='./train_evaluate/results/', 
                save=True, logs=False, count=0, recover=False):
        # default train args
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.model_type = model_type
        self.patience = patience
        self.task = task
        self.num_tasks = num_tasks
        self.normalize = normalize
        self.interval = interval
        self.device = device
        self.metrics = metrics
        self.task_name = task_name
        self.tolerance = tolerance
        self.split = split
        self.results_dir = results_dir
        self.save = save
        self.logs = logs
        self.count = count
        self.recover = recover
        assert len(self.task_name) == num_tasks
        assert self.metrics in ['RMSE', 'MAE', 'multi-MAE', 'ROC-AUC']

        assert self.task in ['regression', 'binary']
        assert type(self.num_epochs) == int 
        assert type(self.batch_size) == int 
        assert type(self.patience) == int


def multitask_mse(train_args):
    def multitask_loss(y_true, y_pred):
        # y_true, y_pred shape = [batch_size, num_tasks]
        loss = 0.0
        for i in range(y_true.shape[1]):
            mse_per_task = (y_true[:, i] - y_pred[:, i]) ** 2
            loss += torch.mean(mse_per_task)*train_args.y_ratio[i]**2
        return loss
    return multitask_loss


def multitask_print(metrics, properties, epoch):
    if properties:
        print(f'epoch' + '\t', end='')
        for property in properties:
            print(f'{property}' + '\t', end='')
        print('\n', end='')  

    print(f'{epoch}' + '\t', end='')
    for metric in metrics:   
        print(f'{metric: .2g}' + '\t', end='')
    print('\n', end='')


def save_results(model, data_loader, train_args):
    y_true, y_pred, smiles = batch_flatten(model, data_loader,
                                    train_args=train_args)
    results = pd.DataFrame(np.concatenate([y_true, y_pred], axis=-1), index=smiles)
    results.to_csv(train_args.results_dir + train_args.model_type + '.csv')