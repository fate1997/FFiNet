from torch import nn
import torch
import numpy as np
from torch_geometric.data import DataLoader

from .train_utils import EarlyStopping, evaluate_score, data2iter, multitask_mse, multitask_print, save_results


def seed_all():
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)


def train(
        model: torch.nn.modules.container.Sequential,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        test_loader: DataLoader, 
        train_args):
    print(f'training on {train_args.device}')
    seed_all()

    early_stopping = EarlyStopping(train_args)
    model.to(train_args.device)
    if train_args.task == 'regression':
        criterion = nn.MSELoss()
    elif train_args.task == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if train_args.normalize == True:
        criterion = multitask_mse(train_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)

    if train_args.recover == True:
        model.load_state_dict(torch.load(train_args.model_save_path + train_args.model_type + '.pt'))

    for epoch in range(train_args.num_epochs):
        model.train()  # get in train mode
        loss_sum = 0
        num_examples = 0
        for i, batch in enumerate(train_loader):
            # forward
            batch = batch.to(train_args.device)
            y = batch.y.reshape((-1, train_args.num_tasks))
            if train_args.normalize == True:
                y = (y - train_args.y_mean) / train_args.y_std
            outputs = model(batch)
            loss = criterion(outputs, y)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_examples += y.shape[0]
            loss_sum += loss.item() * y.shape[0]

        val_metric = evaluate_score(model, eval_loader, train_args)

        if epoch % train_args.interval == 0:
            if train_args.metrics != 'multi-MAE':
                print(f'epoch:{epoch}, loss ={loss_sum / num_examples: .4f}, '
                f'val loss = {val_metric["loss"]:.4f}, '
                f'val {train_args.metrics} = {np.round(val_metric[train_args.metrics], decimals=4)}')
            else:
                multitask_print(val_metric[train_args.metrics], train_args.task_name if epoch==0 else None, epoch)

        
        # early stopping
        min_metrics = np.array(val_metric[train_args.metrics]).mean()
        if train_args.metrics == 'multi-MAE':
            min_metrics = val_metric['loss'].item()

        if train_args.task in ['binary', 'multi-class']:
            min_metrics = -val_metric[train_args.metrics]
        
        early_stopping(min_metrics, test_loader, model)

        if early_stopping.early_stop:
            print('Early stopping!')
            break
    
    model.load_state_dict(torch.load(train_args.model_save_path + train_args.model_type + '.pt'))
    test_metric = evaluate_score(model, test_loader, train_args)
    val_metric = evaluate_score(model, eval_loader, train_args)

    if train_args.metrics != 'multi-MAE':
        print(f'test {train_args.metrics} = {np.round(test_metric[train_args.metrics], decimals=4)}, '
            f'val {train_args.metrics} = {np.round(val_metric[train_args.metrics], decimals=4)}')
    else:
        multitask_print(test_metric[train_args.metrics], train_args.task_name, 'test')
        multitask_print(val_metric[train_args.metrics], None, 'val')

    torch.cuda.empty_cache()
    if train_args.save == True:
        save_results(model, test_loader, train_args)
    return test_metric, val_metric


def evaluate(n, data_path, model_class, model_args, train_args):
    test_metric = []
    val_metric = []

    if train_args.logs:
            print('\n', end='')
            print(f"hidden_dim = {model_args['hidden_dim']}")
            print(f"hidden_layers = {model_args['hidden_layers']}")
            print(f"num_heads = {model_args['num_heads']}")
            print(f"activation = {model_args['activation']}")
            print(f"dropout = {model_args['dropout']}")
            print(f"prediction_hidden_dim = {model_args['prediction_hidden_dim']}")
            print(f"prediction_layers = {model_args['prediction_layers']}")
            print(f"prediction_dropout = {model_args['prediction_dropout']}")
            print(f"batch_size = {train_args.batch_size}")
            print(f"patience = {train_args.patience}")
            print(f"lr = {train_args.lr}")
            print('---------------------------------------------------')

    input_dim = 66
    if train_args.task_name[0] == 'pdbbind': input_dim = 65
    for i in range(n):
        print(f'{i+1}\'s training process')
        model = model_class(
            feature_per_layer=[input_dim] + [model_args['hidden_dim']] * model_args['hidden_layers'], 
            num_heads=model_args['num_heads'], 
            pred_hidden_dim=model_args['prediction_hidden_dim'], 
            pred_dropout=model_args['prediction_dropout'], 
            pred_layers=model_args['prediction_layers'], 
            activation=model_args['activation'], 
            dropout=model_args['dropout'],
            num_tasks=train_args.num_tasks
        )
        
        train_loader, val_loader, test_loader = data2iter(data_path=data_path, seed=i, train_args=train_args)
        train_args.model_type = model_class.__name__ + '_' + data_path.split('\\')[-1].split('.')[0] + f'({str(i)})'
        test, val= train(model, train_loader, val_loader, test_loader, train_args)
        test_metric.append(test[train_args.metrics])
        val_metric.append(val[train_args.metrics])
    test_metric, val_metric = np.array(test_metric), np.array(val_metric)
    
    print("#################################")
    if train_args.metrics != 'multi-MAE':
        print(f'test {train_args.metrics} = {np.mean(test_metric): .4f} ± {np.std(test_metric):.4f}; '
            f'val {train_args.metrics} = {np.mean(val_metric): .4f} ± {np.std(val_metric):.4f}')
    else:
        multitask_print(np.mean(test_metric, axis=0), train_args.task_name, 'test mean')
        multitask_print(np.std(test_metric, axis=0), None, 'test std')
        multitask_print(np.mean(val_metric, axis=0), None, 'val mean')
        multitask_print(np.std(val_metric, axis=0), None, 'val std')
    return np.mean(test_metric), np.mean(val_metric)
