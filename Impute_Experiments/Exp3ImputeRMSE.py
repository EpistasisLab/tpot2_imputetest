import tpot2
import numpy as np
import sklearn.metrics
import sklearn
import argparse
import utils
import autoutils
import time
import sklearn.datasets
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from transformers import RandomForestImputer, GAINImputer
from param_grids import params_SimpleImpute, params_IterativeImpute, params_KNNImpute, params_RandomForestImpute, params_GAINImpute
import openml
import tpot2
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, \
                             roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             f1_score)
from sklearn.model_selection import train_test_split
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


def main():
    # Read in arguements
    parser = argparse.ArgumentParser()
    # number of threads
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    
    #where to save the results/models
    parser.add_argument("-s", "--savepath", default="binary_results", required=False, nargs='?')

    #number of total runs for each experiment
    parser.add_argument("-r", "--num_runs", default=1, required=False, nargs='?')

    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    total_duration = 360000

    
    print('starting loops')
    start = time.time()
    for taskid in ['3764', '3786']:
        fileoutput = '/common/ketrong/AutoImputeExp/tpot2_imputetest/Impute_Experiments/data/'
        csvout = pd.DataFrame(columns=['Exp3ImputeRMSE'], 
                                index=['/tpot2_base_normal_MAR_0.01/','/tpot2_base_normal_MAR_0.1/',
                                    '/tpot2_base_normal_MAR_0.3/','/tpot2_base_normal_MAR_0.5/',
                                        '/tpot2_base_normal_MAR_0.9/','/tpot2_base_normal_MNAR_0.01/',
                                        '/tpot2_base_normal_MNAR_0.1/','/tpot2_base_normal_MNAR_0.3/',
                                        '/tpot2_base_normal_MNAR_0.5/', '/tpot2_base_normal_MNAR_0.9/',
                                        '/tpot2_base_normal_MCAR_0.01/','/tpot2_base_normal_MCAR_0.1/',
                                        '/tpot2_base_normal_MCAR_0.3/','/tpot2_base_normal_MCAR_0.5/',
                                        '/tpot2_base_normal_MCAR_0.9/','/tpot2_base_imputation_MAR_0.01/','/tpot2_base_imputation_MAR_0.1/',
                                    '/tpot2_base_imputation_MAR_0.3/','/tpot2_base_imputation_MAR_0.5/',
                                        '/tpot2_base_imputation_MAR_0.9/','/tpot2_base_imputation_MNAR_0.01/',
                                        '/tpot2_base_imputation_MNAR_0.1/','/tpot2_base_imputation_MNAR_0.3/',
                                        '/tpot2_base_imputation_MNAR_0.5/', '/tpot2_base_imputation_MNAR_0.9/',
                                        '/tpot2_base_imputation_MCAR_0.01/','/tpot2_base_imputation_MCAR_0.1/',
                                        '/tpot2_base_imputation_MCAR_0.3/','/tpot2_base_imputation_MCAR_0.5/', '/tpot2_base_imputation_MCAR_0.9/'])
        #print(csvout)
        for exp in ['/tpot2_base_normal_','/tpot2_base_imputation_']:
            for item in ['MAR_', 'MCAR_', 'MNAR_']:
                for lvl in ['0.01/', '0.1/', '0.3/', '0.5/', '0.9/']:
                    imputepath = '/common/ketrong/AutoImputeExp/tpot2_imputetest/Impute_Experiments/logs/'+ taskid + exp + item + lvl
                    try:
                        with open(imputepath + 'tpot_space_fitted_pipeline.pkl', 'rb') as file:
                            my_run_pipeline = pickle.load(file)
                            print(my_run_pipeline)
                        print(type(my_run_pipeline))
                        print("loading data")
                        levelstr = lvl.replace("/", "")
                        level = float(levelstr)
                        typical = item.replace("_", "")
                        X_train, y_train, X_test, y_test = load_task(imputepath=imputepath, task_id=taskid, preprocess=True)
                        X_train_pandas = pd.DataFrame(X_train)
                        X_test_pandas = pd.DataFrame(X_test)
                        X_train_missing_p, mask_train = utils.add_missing(X_train_pandas, add_missing=level, missing_type=typical)
                        X_test_missing_p, mask_test = utils.add_missing(X_test_pandas, add_missing=level, missing_type=typical)
                        X_train_missing_n = X_train_missing_p.to_numpy()
                        X_test_missing_n = X_test_missing_p.to_numpy()
                        print('fitting')
                        pls_work = my_run_pipeline.fit(X_train, y_train)
                        print('try transform')
                        X_test_transform = pls_work.graph.nodes[pls_work.topo_sorted_nodes[0]]["instance"].transform(X_test_missing_n)
                        print('transform worked')
                        rmse_loss = autoutils.rmse_loss(ori_data=X_test, imputed_data=X_test_transform, data_m=np.multiply(mask_test.to_numpy(),1))
                        print(rmse_loss)
                        csvout.loc[exp+item+lvl] = pd.Series({'Exp3ImputeRMSE': rmse_loss})
                      
                    except:
                        print(taskid+item+lvl+' failed')
                    
        output = csvout.to_csv(fileoutput+taskid+'_3rmse.csv')
        print(taskid + ' Complete')

    stop = time.time()
    duration = stop - start
    print('full run takes')
    print(duration/3600)
    print('hours')

def load_task(imputepath, task_id, preprocess=True):
    
    cached_data_path = imputepath + f"data/{task_id}_{preprocess}.pkl"
    print(cached_data_path)
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        #kwargs = {'force_refresh_cache': True}
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

            '''
            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            '''

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists(imputepath + f"data/"):
                os.makedirs(imputepath + f"data/")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test

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

if __name__ == '__main__':
    main()
    print("DONE")