import tpot2
import numpy as np
import optuna
import sklearn

def params_SimpleImpute(trial, name=None):
    params = {}
    params['strategy'] = trial.suggest_categorical('strategy', ['mean', 'median', 'most_frequent', 'constant'])
    param_grid = {
        'strategy': params['strategy']
    }
    return param_grid

def params_IterativeImpute(trial, name=None):
    params = {}
    params['estimator'] = trial.suggest_categorical('estimator', ['Bayesian', 'RFR', 'Ridge', 'KNN'])
    params['sample_posterior'] = trial.suggest_categorical('sample_posterior', [True, False])
    params['initial_strategy'] = trial.suggest_categorical('initial_strategy', ['mean', 'median', 'most_frequent', 'constant'])
    params['n_nearest_features'] = None
    params['imputation_order'] = trial.suggest_categorical('imputation_order', ['ascending', 'descending', 'roman', 'arabic', 'random'])
    match params['estimator']: #do hyperparam search here too?
        case 'Bayesian':
            params['estimator'] = sklearn.linear_model.BayesianRidge()
        case 'RFR':
            params['estimator'] = sklearn.ensemble.RandomForestRegressor()
        case 'Ridge':
            params['estimator'] = sklearn.linear_model.Ridge()
        case 'KNN':
            params['estimator'] = sklearn.neighbors.KNeighborsRegressor()
    param_grid = {
        'estimator': params['estimator'],
        'sample_posterior': params['sample_posterior'],
        'initial_strategy': params['initial_strategy'],
        'n_nearest_features': params['n_nearest_features'],
        'imputation_order': params['imputation_order']
    }
    return param_grid

def params_KNNImpute(trial, name=None):
    params = {}
    params['n_nearest_features'] = None
    params['weights'] = trial.suggest_categorical('weights', ['uniform', 'distance'])
    params['keep_empty_features'] = trial.suggest_categorical('keep_empty_features', [True, False])
    param_grid = {
      'n_neighbors': params['n_nearest_features'],
      'weights': params['weights'],
      'add_indicator': False,
      'keep_empty_features': params['keep_empty_features'],
    }
    return param_grid

def params_RandomForestImpute(trial, name=None):
    params = {}
    