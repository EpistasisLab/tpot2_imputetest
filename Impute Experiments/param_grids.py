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
    params['max_iter'] = trial.suggest_int('max_iter', 1, 100, step = 1, log = True)
    params['decreasing'] = trial.suggest_categorical('decreasing', [True, False])
    params['n_estimators'] = trial.suggest_int('max_iter', 50, 200, step = 1, log=True)
    params['max_depth'] = None
    params['min_samples_split'] = trial.suggest_float('min_samples_split', 0.0, 1.0, step = 0.1)
    params['min_samples_leaf'] = trial.suggest_float('min_samples_leaf', 0.0, 1.0, step = 0.1)
    params['max_features'] = trial.suggest_float('max_features', 0.0, 1.0, step = 0.1)
    params['max_leaf_nodes'] = None
    params['bootstrap'] = trial.suggest_categorical('bootstrap', [True, False])
    params['oob_score'] = trial.suggest_categorical('oob_score', [True, False])
    params['warm_start'] = trial.suggest_categorical('warm_start', [True, False])
    params['class_weight'] = None
    param_grid = {
        'max_iter': params['max_iter'],
        'decreasing': params['decreasing'],
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'max_features': params['max_features'], 
        'max_leaf_nodes': params['max_leaf_nodes'],
        'bootstrap': params['bootstrap'],
        'oob_score': params['oob_score'],
        'warm_start': params['warm_start'],
        'class_weight': params['class_weight']
    }
    return param_grid

def params_GAINImpute(trial, name=None):
    params = {}
    params['batch_size'] = trial.suggest_int('batch_size', 1, 1000, log=True)
    params['hint_rate'] = trial.suggest_float('hint_rate', 0.001, 1, log=True)
    params['alpha'] = trial.suggest_int('alpha', 0, 100, step = 1)
    params['iterations'] = trial.suggest_int('iterations', 1, 100000, log=True)
    param_grid = { 
        'batch_size': params['batch_size'],
        'hint_rate': params['hint_rate'],
        'alpha': params['alpha'],
        'iterations': params['iterations']
    }
    return param_grid

