import openml
import tpot2
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, \
                             roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             f1_score)
import traceback
import dill as pickle
import os
import time
import tpot
import openml
import tpot2
import sklearn.datasets
import numpy as np
import time
import random
import sklearn.model_selection
import torch
from scipy import optimize
import pandas as pd
import autoimpute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from transformers import RandomForestImputer, GAINImputer
from param_grids import params_SimpleImpute, params_IterativeImpute, params_KNNImpute, params_RandomForestImpute, params_GAINImpute

n_jobs = 48
total_duration = 360000
'''
imputation_config_dict = {
                SimpleImputer: params_SimpleImpute, 
                IterativeImputer: params_IterativeImpute,
                KNNImputer: params_KNNImpute,
                RandomForestImputer: params_RandomForestImpute,
                GAINImputer: params_GAINImpute
    }

simple_config_dict = {
            SimpleImputer: params_SimpleImpute
}
simple_params = {
                'root_config_dict':simple_config_dict,
                'leaf_config_dict': None,
                'inner_config_dict':None,
                'max_size' : 1,
                'linear_pipeline' : True
                }

imputation_params =  {
            'root_config_dict':imputation_config_dict,
            'leaf_config_dict': None,
            'inner_config_dict':None,
            'max_size' : 1,
            'linear_pipeline' : True
            }
'''
normal_params =  {
                'root_config_dict':["classifiers"],
                'leaf_config_dict': None,
                'inner_config_dict': ["selectors", "transformers"],
                'max_size' : np.inf,
                'linear_pipeline' : True,

                'scorers':['neg_log_loss', tpot2.objectives.complexity_scorer],
                'scorers_weights':[1,-1],
                'other_objective_functions':[],
                'other_objective_functions_weights':[],
                
                'population_size' : n_jobs,
                'survival_percentage':1, 
                'initial_population_size' : n_jobs,
                'generations' : 75, 
                'n_jobs':n_jobs,
                'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                'verbose':5, 
                'max_time_seconds': total_duration,
                'max_eval_time_seconds':60*10, 

                'crossover_probability':.10,
                'mutate_probability':.90,
                'mutate_then_crossover_probability':0,
                'crossover_then_mutate_probability':0,


                'memory_limit':None,  
                'preprocessing':False,
                'classification' : True,
                }
'''
imputation_params_and_normal_params = {
                'root_config_dict': {"Recursive" : normal_params},
                'leaf_config_dict': {"Recursive" : imputation_params},
                'inner_config_dict': None,
                'max_size' : np.inf,
                'linear_pipeline' : True,

                #'scorers':['neg_log_loss', tpot2.objectives.complexity_scorer],
                'scorers_weights':[1,-1],
                'other_objective_functions':[],
                'other_objective_functions_weights':[],
                
                'population_size' : n_jobs,
                'survival_percentage':1, 
                'initial_population_size' : n_jobs,
                'generations' : 75, 
                'n_jobs':n_jobs,
                'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                'verbose':5, 
                'max_time_seconds': total_duration,
                'max_eval_time_seconds':60*10, 

                'crossover_probability':.10,
                'mutate_probability':.90,
                'mutate_then_crossover_probability':0,
                'crossover_then_mutate_probability':0,


                'memory_limit':None,  
                'preprocessing':False,
                'classification' : True,
            }

simple_and_normal_params = {
                'root_config_dict': {"Recursive" : normal_params},
                'leaf_config_dict': {"Recursive" : simple_params},
                'inner_config_dict': None,
                'max_size' : np.inf,
                'linear_pipeline' : True,

                'scorers':['neg_log_loss', tpot2.objectives.complexity_scorer],
                'scorers_weights':[1,-1],
                'other_objective_functions':[],
                'other_objective_functions_weights':[],
                
                'population_size' : n_jobs,
                'survival_percentage':1, 
                'initial_population_size' : n_jobs,
                'generations' : 75, 
                'n_jobs':n_jobs,
                'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                'verbose':5, 
                'max_time_seconds': total_duration,
                'max_eval_time_seconds':60*10, 

                'crossover_probability':.10,
                'mutate_probability':.90,
                'mutate_then_crossover_probability':0,
                'crossover_then_mutate_probability':0,


                'memory_limit':None,  
                'preprocessing':False,
                'classification' : True,

}
'''


def score(est, X, y):

    try:
        this_auroc_score = sklearn.metrics.get_scorer("roc_auc_ovr")(est, X, y)
    except:
        y_preds = est.predict(X)
        y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
        this_auroc_score = roc_auc_score(y, y_preds_onehot, multi_class="ovr")
    
    try:
        this_logloss = sklearn.metrics.get_scorer("neg_log_loss")(est, X, y)*-1
    except:
        y_preds = est.predict(X)
        y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
        this_logloss = log_loss(y, y_preds_onehot)

    this_accuracy_score = sklearn.metrics.get_scorer("accuracy")(est, X, y)
    this_balanced_accuracy_score = sklearn.metrics.get_scorer("balanced_accuracy")(est, X, y)


    return { "auroc": this_auroc_score,
            "accuracy": this_accuracy_score,
            "balanced_accuracy": this_balanced_accuracy_score,
            "logloss": this_logloss,
    }



#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id, preprocess=True):
    
    cached_data_path = f"tpot2_imputetest/Impute_Experiments/data/{task_id}_{preprocess}.pkl"
    print(cached_data_path)
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        task = openml.tasks.get_task(task_id)
    
    
        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        if preprocess:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)

            
            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("tpot2_imputetest/Impute_Experiments/data"):
                os.makedirs("tpot2_imputetest/Impute_Experiments/data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test


def loop_through_tasks(experiments, task_id_lists, base_save_folder, num_runs):
    match num_runs: 
            case 1: 
                level = 0.01
                type = 'MCAR'
            case 2: 
                level = 0.1
                type = 'MCAR'
            case 3: 
                level = 0.3
                type = 'MCAR'
            case 4: 
                level = 0.5
                type = 'MCAR'
            case 5: 
                level = 0.9
                type = 'MCAR'
            case 6: 
                level = 0.01
                type = 'MAR'
            case 7: 
                level = 0.1
                type = 'MAR'
            case 8: 
                level = 0.3
                type = 'MAR'
            case 9: 
                level = 0.5
                type = 'MAR'
            case 10: 
                level = 0.9
                type = 'MAR'
            case 11: 
                level = 0.01
                type = 'MNAR'
            case 12: 
                level = 0.1
                type = 'MNAR'
            case 13: 
                level = 0.3
                type = 'MNAR'
            case 14: 
                level = 0.5
                type = 'MNAR'
            case 15: 
                level = 0.9
                type = 'MNAR'

    for taskid in task_id_lists:
        for exp in experiments:
            levelstr = str(level)
            save_folder = f"{base_save_folder}/{exp['exp_name']}_{taskid}_{levelstr}_{type}"
            checkpoint_folder = f"{base_save_folder}/checkpoint/{exp['exp_name']}_{taskid}_{levelstr}_{type}"
            time.sleep(random.random()*5)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            else:
                continue

            time.sleep(random.random()*5)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            else:
                continue

            print("working on ")
            print(save_folder)

            start = time.time()
            time.sleep(random.random()*5)
            duration = time.time() - start
            print(duration)

            try: 
                print("loading data")
                X_train, y_train, X_test, y_test = load_task(taskid, preprocess=True)
                

                print("running experiment 1/3 - Does large hyperparameter space improve reconstruction accuracy over simple")
                X_train_pandas = pd.DataFrame(X_train)
                X_test_pandas = pd.DataFrame(X_test)
                '''
                #Simple Impute 
                SimpleImputeSpace = autoimpute.AutoImputer(added_missing=level, missing_type=type, model_names=['SimpleImputer'], n_jobs=48, show_progress=False, random_state=num_runs)
                SimpleImputeSpace.fit(X_train_pandas)
                print('simple fit')
                simple_impute = SimpleImputeSpace.transform(X_test_pandas)
                print('simple transform')
                simple_rmse = SimpleImputeSpace.study.best_trial.value
                simple_space = SimpleImputeSpace.study.best_trial.params
                simple_impute = simple_impute.to_numpy()
                print(simple_rmse)
                print(simple_rmse)
                #Auto Impute 
                AutoImputeSpace = autoimpute.AutoImputer(added_missing=level, missing_type=type, model_names=['SimpleImputer', 'IterativeImputer', 'KNNImputer', 'GAIN', 'RandomForestImputer'], n_jobs=48, show_progress=False, random_state=num_runs)
                AutoImputeSpace.fit(X_train_pandas)
                print('auto fit')
                auto_impute = AutoImputeSpace.transform(X_test_pandas)
                print('auto transform')
                auto_rmse = AutoImputeSpace.study.best_trial.value
                auto_space = AutoImputeSpace.study.best_trial.params
                auto_impute = auto_impute.to_numpy()
                print(auto_rmse)
                print(auto_space)

                all_scores = {}
                all_scores['simple_impute_rmse'] = simple_rmse
                all_scores['auto_impute_rmse'] = auto_rmse
                all_scores['simple_impute_space'] = simple_space
                all_scores['auto_impute_space'] = auto_space

                print("running experiment 2/3 - Does reconstruction give good automl predictions")
                #this section trains off of original train data, and then tests on the original, the simpleimputed,
                #  and the autoimpute test data. This section uses the normal params since it is checking just for predictive preformance, 
                # not the role of various imputers in the tpot optimization space. 

                exp['params']['cv'] = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=num_runs)
                exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
                est = exp['automl'](**normal_params)

                print('Start est fit')
                start = time.time()
                est.fit(X_train, y_train)
                duration = time.time() - start
                print('Fitted')
                if type(est) is tpot.TPOTClassifier:
                    est.classes_ = est.fitted_pipeline_.classes_
                print('score start')
                train_score = score(est, X_train, y_train)
                ori_test_score = score(est, X_test, y_test)
                simple_test_score = score(est, simple_impute, y_test)
                auto_test_score = score(est, auto_impute, y_test)
                print('score end')
                train_score = {f"train_{k}": v for k, v in train_score.items()}
                all_scores.update(train_score)
                all_scores.update(ori_test_score)
                all_scores.update(simple_test_score)
                all_scores.update(auto_test_score)
                
                all_scores["start"] = start
                all_scores["taskid"] = taskid
                all_scores["level"] = level
                all_scores["type"] = type
                all_scores["exp_name"] = 'Imputed_Predictive_Capacity'
                #all_scores["name"] = openml.datasets.get_dataset(openml.tasks.get_task(taskid).dataset_id).name
                all_scores["duration"] = duration
                all_scores["run"] = num_runs
                #'''
                print("starting impute modules")
                X_train_missing, mask_train = add_missing(X_train_pandas, add_missing=level, missing_type=type)
                X_test_missing, mask_test = add_missing(X_test_pandas, add_missing=level, missing_type=type)
                X_train_missing = X_train_missing.to_numpy()
                X_test_missing = X_test_missing.to_numpy()


                exp['params']['cv'] = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=num_runs)
                exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
                tpot_space = exp['automl'](**exp['params'])
                print('Start tpot fit')
                start = time.time()
                tpot_space.fit(X_train_missing, y_train)
                duration = time.time() - start
                print('Fitted')
                if type(tpot_space) is tpot.TPOTClassifier:
                    tpot_space.classes_ = tpot_space.fitted_pipeline_.classes_
                print('score start')
                train_score = score(tpot_space, X_train_missing, y_train)
                test_score = score(tpot_space, X_test_missing, y_test)
                print('score end')
                tpot_space_scores = {}
                train_score = {f"train_{k}": v for k, v in train_score.items()}
                tpot_space_scores.update(train_score)
                tpot_space_scores.update(test_score)

                
                tpot_space_scores["start"] = start
                tpot_space_scores["taskid"] = taskid
                tpot_space_scores["exp_name"] = exp['exp_name']
                #tpot_space_scores["name"] = openml.datasets.get_dataset(openml.tasks.get_task(taskid).dataset_id).name
                tpot_space_scores["duration"] = duration
                tpot_space_scores["run"] = num_runs

                if type(est) is tpot2.TPOTClassifier or type(est) is tpot2.TPOTEstimator or type(est) is  tpot2.TPOTEstimatorSteadyState:
                    with open(f"{save_folder}/evaluated_individuals.pkl", "wb") as f:
                        pickle.dump(est.evaluated_individuals, f)

                with open(f"{save_folder}/fitted_pipeline.pkl", "wb") as f:
                    pickle.dump(est.fitted_pipeline_, f)

                with open(f"{save_folder}/all_scores.pkl", "wb") as f:
                    pickle.dump(all_scores, f)

                if type(tpot_space) is tpot2.TPOTClassifier or type(tpot_space) is tpot2.TPOTEstimator or type(tpot_space) is  tpot2.TPOTEstimatorSteadyState:
                    with open(f"{save_folder}/tpot_space_evaluated_individuals.pkl", "wb") as f:
                        pickle.dump(tpot_space.evaluated_individuals, f)

                with open(f"{save_folder}/tpot_space_fitted_pipeline.pkl", "wb") as f:
                    pickle.dump(tpot_space.fitted_pipeline_, f)

                with open(f"{save_folder}/tpot_space_scores.pkl", "wb") as f:
                    pickle.dump(tpot_space_scores, f)
                #'''
                return
            except Exception as e:
                trace =  traceback.format_exc() 
                pipeline_failure_dict = {"taskid": taskid, "exp_name": exp['exp_name'], "run": num_runs, "error": str(e), "trace": trace, "level": level, "type": type}
                print("failed on ")
                print(save_folder)
                print(e)
                print(trace)

                with open(f"{save_folder}/failed.pkl", "wb") as f:
                    pickle.dump(pipeline_failure_dict, f)

                return
        print(taskid)
        print('finished')
    print("all finished")


### Additional Stuff GKetron Added
def add_missing(X, add_missing = 0.05, missing_type = 'MAR'):
    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X)
    missing_mask = X
    missing_mask = missing_mask.mask(missing_mask.isna(), True)
    missing_mask = missing_mask.mask(missing_mask.notna(), False)
    X = X.mask(X.isna(), 0)
    T = torch.tensor(X.to_numpy())

    match missing_type:
        case 'MAR':
            out = MAR(T, [add_missing])
        case 'MCAR':
            out = MCAR(T, [add_missing])
        case 'MNAR':
            out = MNAR_mask_logistic(T, [add_missing])

    masked_set = pd.DataFrame(out['Mask'].numpy())
    masked_set = masked_set.mask((missing_mask | masked_set), True)
    masked_set.columns = X.columns.values
    #masked_set = masked_set.to_numpy()

    missing_set = pd.DataFrame(out['Missing'].numpy())
    missing_set.columns = X.columns.values
    #missing_set = missing_set.to_numpy()

    return missing_set, masked_set

"""BEYOND THIS POINT WRITTEN BY Aude Sportisse, Marine Le Morvan and Boris Muzellec - https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values"""

def MCAR(X, p_miss):
    out = {'X': X.double()}
    for p in p_miss: 
        mask = (torch.rand(X.shape) < p).double()
        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        model_name = 'Missing'
        mask_name = 'Mask'
        out[model_name] = X_nas
        out[mask_name] = mask
    return out

def MAR(X,p_miss,p_obs=0.5):
    out = {'X': X.double()}
    for p in p_miss:
        n, d = X.shape
        mask = torch.zeros(n, d).bool()
        num_no_missing = max(int(p_obs * d), 1)
        num_missing = d - num_no_missing
        obs_samples = np.random.choice(d, num_no_missing, replace=False)
        copy_samples = np.array([i for i in range(d) if i not in obs_samples])
        len_obs = len(obs_samples)
        len_na = len(copy_samples)
        coeffs = torch.randn(len_obs, len_na).double()
        Wx = X[:, obs_samples].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
        coeffs.double()
        len_obs, len_na = coeffs.shape
        intercepts = torch.zeros(len_na)
        for j in range(len_na):
            def f(x):
                return torch.sigmoid(X[:, obs_samples].mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
        ps = torch.sigmoid(X[:, obs_samples].mm(coeffs) + intercepts)
        ber = torch.rand(n, len_na)
        mask[:, copy_samples] = ber < ps
        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        model_name = 'Missing'
        mask_name = 'Mask'
        out[model_name] = X_nas
        out[mask_name] = mask
    return out

def MNAR_mask_logistic(X, p_miss, p_params =.5, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """
    out = {'X_init_MNAR': X.double()}
    for p in p_miss: 
        n, d = X.shape
        to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = torch.from_numpy(X)
        mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)
        d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
        d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model
        ### Sample variables that will be parameters for the logistic regression:
        idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)
        ### Other variables will have NA proportions selected by a logistic model
        ### The parameters of this logistic model are random.
        ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
        len_obs = len(idxs_params)
        len_na = len(idxs_nas)
        coeffs = torch.randn(len_obs, len_na).double()
        Wx = X[:, idxs_params].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
        coeffs.double()
        ### Pick the intercepts to have a desired amount of missing values
        len_obs, len_na = coeffs.shape
        intercepts = torch.zeros(len_na)
        for j in range(len_na):
            def f(x):
                return torch.sigmoid(X[:, idxs_params].mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
        ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)
        ber = torch.rand(n, d_na)
        mask[:, idxs_nas] = ber < ps
        ## If the inputs of the logistic model are excluded from MNAR missingness,
        ## mask some values used in the logistic model at random.
        ## This makes the missingness of other variables potentially dependent on masked values
        if exclude_inputs:
            mask[:, idxs_params] = torch.rand(n, d_params) < p
        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan
        model_name = 'Missing'
        mask_name = 'Mask'
        out[model_name] = X_nas
        out[mask_name] = mask
    return out

