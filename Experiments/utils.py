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
import optuna
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from gainfunction import GAINImputer
from IPython.display import clear_output

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
    
    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
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
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test


def loop_through_tasks(experiments, task_id_lists, base_save_folder):
    for taskid in task_id_lists:
        for level in [0.01, 0.1, 0.3, 0.5, 0.9]:
            for type in ['MCAR', 'MNAR', 'MAR']:
                for exp in experiments:
                    save_folder = f"{base_save_folder}/{exp['exp_name']}_{taskid}_{level}_{type}"
                    time.sleep(random.random()*5)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    else:
                        continue

                    print("working on ")
                    print(save_folder)

                    try: 
                        print("loading data")
                        X_train, y_train, X_test, y_test = load_task(taskid, preprocess=True)
                        
                        print("adding missingness")
                        missing_train, mask_train = add_missing(X=X_train, add_missing=level, missing_type=type)
                        missing_test, mask_test = add_missing(X=X_train, add_missing=level, missing_type=type)



                        print("starting ml")
                        exp['params']['cv'] = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=run)
                        exp['params']['periodic_checkpoint_folder'] = f"/home/ribeirop/common/Projects/tpot_digen_paper1/tpot2_paper_1/checkpoint/{exp['exp_name']}_{taskid}_{run}"
                        est = exp['automl'](**exp['params'])

                        start = time.time()
                        est.fit(X_train, y_train)
                        duration = time.time() - start
                        
                        if type(est) is tpot.TPOTClassifier:
                            est.classes_ = est.fitted_pipeline_.classes_

                        train_score = score(est, X_train, y_train)
                        test_score = score(est, X_test, y_test)

                        all_scores = {}
                        train_score = {f"train_{k}": v for k, v in train_score.items()}
                        all_scores.update(train_score)
                        all_scores.update(test_score)

                        
                        all_scores["start"] = start
                        all_scores["taskid"] = taskid
                        all_scores["exp_name"] = exp['exp_name']
                        #all_scores["name"] = openml.datasets.get_dataset(openml.tasks.get_task(taskid).dataset_id).name
                        all_scores["duration"] = duration
                        all_scores["run"] = run

                        if type(est) is tpot2.TPOTClassifier or type(est) is tpot2.TPOTEstimator or type(est) is  tpot2.TPOTEstimatorSteadyState:
                            with open(f"{save_folder}/evaluated_individuals.pkl", "wb") as f:
                                pickle.dump(est.evaluated_individuals, f)

                        
                        with open(f"{save_folder}/fitted_pipeline.pkl", "wb") as f:
                            pickle.dump(est.fitted_pipeline_, f)


                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(all_scores, f)

                        return
                    except Exception as e:
                        trace =  traceback.format_exc() 
                        pipeline_failure_dict = {"taskid": taskid, "exp_name": exp['exp_name'], "run": run, "error": str(e), "trace": trace}
                        print("failed on ")
                        print(save_folder)
                        print(e)
                        print(trace)

                        with open(f"{save_folder}/failed.pkl", "wb") as f:
                            pickle.dump(pipeline_failure_dict, f)

                        return
    
    print("all finished")

def add_missing(X, add_missing = 0.05, missing_type = 'MAR'):
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
    masked_set = masked_set.to_numpy()

    missing_set = pd.DataFrame(out['Missing'].numpy())
    missing_set.columns = X.columns.values
    missing_set = missing_set.to_numpy()

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

def trial_suggestion(trial: optuna.trial.Trial, model_names,column_len,random_state):
    model_name = trial.suggest_categorical('model_name', model_names)# Model names set up to run on multiple or individual models. Options include: 'SimpleImputer' , 'IterativeImputer','KNNImputer', 'VAEImputer', 'GAIN', 'Opt.SVM', 'Opt.Trees'.  Not yet working: 'DLImputer', Random Forest Imputer 
    match model_name:
      case 'SimpleImputer':
        my_params = SimpleSuggest(trial, model_name)  #Takes data from each column to input a value for all missing data. 
      case 'IterativeImputer':
        my_params = IterativeSuggest(trial,model_name, column_len, random_state) #Uses the dependence between columns in the data set to predict for the one column. Predictions occur through a variety of strategies.
      case 'KNNImputer':
        my_params = KNNSuggest(trial,model_name, column_len) #uses nearest neighbors to predict missing values with k-neighbors in n-dimensional space with known values.
      case 'GAIN':
        my_params = GAINSuggest(trial,model_name) #Uses a generative adversarial network model to predict values. 
    return my_params
  
def MyModel(random_state, **params):
    model_name = params['model_name']
    match model_name:
        case 'SimpleImputer':
            my_model = SimpleImputer(
                strategy = params['strategy']
                )
        case 'IterativeImputer':
            my_model = IterativeImputer(
                estimator=params['estimator'],
                sample_posterior=params['sample_posterior'],
                initial_strategy=params['initial_strategy'],
                n_nearest_features=params['n_nearest_features'],
                imputation_order=params['imputation_order'],
                random_state=random_state
                )
        case 'KNNImputer':
            my_model = KNNImputer(
                n_neighbors=params['n_neighbors'],
                weights=params['weights'],
                add_indicator=False,
                keep_empty_features=params['keep_empty_features'],
                )
        case 'GAIN':
            my_model = GAINImputer(
                **params
                )
    return my_model

def score(trial: optuna.trial.Trial, splitting, my_model, X: pd.DataFrame, missing_set: pd.DataFrame, masked_set:pd.DataFrame):
    avg_cv_rmse = []
    try: 
        for i, (train_index, test_index) in enumerate(splitting.split(X)): ## Creates test and train splits based on rows selected for the most recent cv split. Splits start as pandas data frames and may switch to numpy depending on imputer model inputs.
            missing_train, missing_test, X_test, masked_test = missing_set.iloc[train_index], missing_set.iloc[test_index], X.iloc[test_index], masked_set.iloc[test_index]
            my_model.fit(missing_train)
            imputed = my_model.transform(missing_test)
            if 'numpy' in str(type(imputed)):
                imputed = pd.DataFrame(imputed, columns=missing_set.columns.values)
            rmse_val = rmse_loss(ori_data=X_test.to_numpy(), imputed_data=imputed.to_numpy(), data_m=np.multiply(masked_test.to_numpy(),1))
            avg_cv_rmse.append(rmse_val)
        cv_rmse = sum(avg_cv_rmse)/float(len(avg_cv_rmse))
    except:
        cv_rmse = np.inf
    trial.set_user_attr('cv_rmse', cv_rmse)
    clear_output()
    return cv_rmse

def SimpleSuggest(trial, model_name):
    strategy = trial.suggest_categorical('strategy', ['mean', 'median', 'most_frequent', 'constant'])
    my_params = {
        'model_name': model_name,
        'strategy': strategy
        }
    return my_params
  
def IterativeSuggest(trial, model_name, column_len, random_state):
    estimator = trial.suggest_categorical('estimator', ['Bayesian', 'RFR', 'Ridge', 'KNN']) #, 'RFR' 'Nystroem',
    sample_posterior = trial.suggest_categorical('sample_posterior', [True, False])
    initial_strategy = trial.suggest_categorical('initial_strategy', ['mean', 'median', 'most_frequent', 'constant'])
    n_nearest_features = trial.suggest_int('n_nearest_features',1, column_len, step=2)
    imputation_order = trial.suggest_categorical('imputation_order', ['ascending', 'descending', 'roman', 'arabic', 'random'])
    match estimator: #do hyperparam search here too?
        case 'Bayesian':
            estimator = sklearn.linear_model.BayesianRidge()
        case 'RFR':
            estimator = RandomForestRegressor()
        case 'Ridge':
            estimator = sklearn.linear_model.Ridge()
        case 'KNN':
            estimator = sklearn.neighbors.KNeighborsRegressor()
    my_params = {
        'model_name': model_name,
        'estimator': estimator,
        'sample_posterior': sample_posterior,
        'initial_strategy': initial_strategy,
        'n_nearest_features': n_nearest_features,
        'imputation_order': imputation_order,
        'random_state': random_state
    }
    return my_params

def KNNSuggest(trial, model_name, column_len):
    n_neighbors = trial.suggest_int('n_neighbors', 1, column_len, step = 1)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    keep_empty_features = trial.suggest_categorical('keep_empty_features', [True, False])
    my_params = {
      'model_name': model_name,
      'n_neighbors': n_neighbors,
      'weights': weights,
      'add_indicator': False,
      'keep_empty_features': keep_empty_features,
    }
    return my_params
  
def GAINSuggest(trial, model_name):
    batch_size = trial.suggest_int('batch_size', 1, 1000, log=True)
    hint_rate = trial.suggest_float('hint_rate', 0.001, 1, log=True)
    alpha = trial.suggest_int('alpha', 0, 100, step = 1)
    iterations = trial.suggest_int('iterations', 1, 100000, log=True)
    my_params = {
      'model_name': model_name, 
      'batch_size': batch_size,
      'hint_rate': hint_rate,
      'alpha': alpha,
      'iterations': iterations
      }
    return my_params

def OptSvmSuggest(trial, model_name, random_state):
    svm_gamma = trial.suggest_float('svm_gamma', 0.001, 10, log = True)
    svm_epsilon = trial.suggest_float('svm_epsilon', 0.001, 1, log=True)
    max_iter = trial.suggest_int('max_iter', 1, 200, log=True)
    cluster = trial.suggest_categorical('cluster', [True, False])
    my_params = {
        'model_name': model_name,
        'svm_gamma': svm_gamma,
        'svm_epsilon': svm_epsilon,
        'max_iter': max_iter,
        'cluster': cluster,
        'random_state': random_state
    }
    return my_params
  
def OptTreesSuggest(trial, model_name, random_state):
    max_iter = trial.suggest_int('max_iter', 1, 200, log = True)
    cluster = trial.suggest_categorical('cluster', [True, False])
    my_params = {
        'model_name': model_name,
        'max_iter': max_iter,
        'cluster' : cluster,
        'random_state': random_state
    }
    return my_params
  
def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data
    Args:
        - ori_data: original data without missing values
        - imputed_data: imputed data
        - data_m: indicator matrix for missingness    
    Returns:
        - rmse: Root Mean Squared Error
    '''
    #ori_data, norm_parameters = normalization(ori_data)
    #imputed_data, _ = normalization(imputed_data, norm_parameters)
    # Only for missing values
    nominator = np.nansum(((data_m) * ori_data - (data_m) * imputed_data)**2)
    #print(nominator)
    denominator = np.sum(data_m)
    rmse = np.sqrt(nominator/float(denominator))
    return rmse

""" BEYOND THIS POINT WRITTEN BY Piotr Slomka - https://www.nature.com/articles/s41598-021-93651-5.pdf"""

def convert_col(pd_series,keep_missing=True):
    df = pd.DataFrame(pd_series)
    new_df =  df.stack().str.get_dummies(sep=',')
    new_df.columns = new_df.columns.str.strip()
    if keep_missing and 'nan' in new_df.columns: #this means there were missing values
        new_df.loc[new_df['nan']==1] = np.nan
        new_df.drop('nan', inplace=True,axis=1)
    new_df = new_df.stack().groupby(level=[0,1,2]).sum().unstack(level=[1,2])
    return new_df

#One hot encodes all categorical features. (if continuous included, each value will be counted as a discrete category)
def convert_all_cols(df,features_to_one_hot,keep_missing=True):
    new_df = pd.concat([convert_col(df[feature],keep_missing) for feature in features_to_one_hot],axis=1)
    return pd.concat([df.drop(features_to_one_hot, axis = 1),new_df],axis=1)

def one_hot_encode(spect_df,keep_missing=True):
    categorical_feature_names = spect_df.select_dtypes(include=object).columns
    numerical_feature_names = spect_df.select_dtypes(include=float).columns
    spect_df = convert_all_cols(spect_df,categorical_feature_names,keep_missing=keep_missing)
    #change binary ints into floats
    spect_df = spect_df.astype(float)
    #change column names from tuples to strings for the imputer to work better
    change_column_name_from_tuple_to_string = lambda cname: '__'.join(cname) if type(cname)==tuple else cname
    spect_df.columns = spect_df.columns.map(change_column_name_from_tuple_to_string)
    return spect_df