from hyperopt import fmin, hp, tpe, Trials, rand
import torch.nn as nn
from train_graph import evaluate
import numpy as np

SEARCH_SPACE = {
    'hidden_dim': hp.choice('hidden_dim', [16, 32, 64, 96]), 
    'hidden_layers': hp.choice('hidden_layers', [1, 2, 3, 4, 5]),
    'num_heads': hp.choice('num_heads', [4, 8, 12]),
    'activation': hp.choice('activation', [nn.PReLU(), nn.ELU()]), 
    'dropout': hp.choice('dropout', [0.1, 0.2, 0.3]), 
    'prediction_layers': hp.choice('prediction_layers', [1, 2]),
    'prediction_dropout': hp.choice('prediction_dropout', [0.1, 0.2]),
    'prediction_hidden_dim': hp.choice('prediction_hidden_dim', [256, 512]),
    'batch_size': hp.choice('batch_size', [128]), 
    'patience': hp.choice('patience', [500]),
    'lr': hp.choice('lr', [5e-4, 2e-3]),
}


def run_opt(num_iters, seed, data_path, model_class, train_args, higher_is_better=False):
    
    results = []
    def objective(hyperparams):
        
        train_args.batch_size = hyperparams['batch_size']
        train_args.patience = hyperparams['patience']
        train_args.lr = hyperparams['lr']

        test_metric, val_metric = evaluate(3, 
                                    data_path=data_path, 
                                    model_class=model_class, 
                                    model_args=hyperparams, 
                                    train_args=train_args
                                    )
        
        results.append({
            'test_metrics': test_metric, 
            'val_metrics': val_metric, 
            'hyperparams': hyperparams
        })
        return val_metric * (-1 if higher_is_better else 1)

    trials = Trials()
    fmin(objective, SEARCH_SPACE, algo=tpe.suggest, max_evals=num_iters, rstate=np.random.RandomState(seed), trials=trials)

    best_result = min(results, key=lambda result: result['val_metrics'])

    return best_result['test_metrics']