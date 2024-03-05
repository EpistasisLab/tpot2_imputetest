import tpot2
import numpy as np
import sklearn.metrics
import sklearn
import argparse
import utils
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
    for taskid in ['2306', '2309', '2288', '2289', '2307', '359935','7320', '7323', '233211', '359938', '317615']:
        fileoutput = '/common/ketrong/AutoImputeExp/tpot2_imputetest/Impute_Experiments/regression_data/'
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
                        print("loading data")
                        levelstr = lvl.replace("/", "")
                        level = float(levelstr)
                        type = item.replace("_", "")
                        X_train, y_train, X_test, y_test = load_task(imputepath=imputepath, task_id=taskid, preprocess=True)
                        print(y_train)
                        X_train_pandas = pd.DataFrame(X_train)
                        print(X_train_pandas)
                        X_test_pandas = pd.DataFrame(X_test)
                        print(X_test_pandas)
                        X_train_missing_p, mask_train = utils.add_missing(X_train_pandas, add_missing=level, missing_type=type)
                        X_test_missing_p, mask_test = utils.add_missing(X_test_pandas, add_missing=level, missing_type=type)
                        X_train_missing_n = X_train_missing_p.to_numpy()
                        X_test_missing_n = X_test_missing_p.to_numpy()

                        




                        csvout.loc[exp+item+lvl] = pd.Series({'Exp3ImputeRMSE': )
                        
                    except:
                        print(taskid+item+lvl+' failed')
        output = csvout.to_csv(fileoutput+taskid+'.csv')

    stop = time.time()
    duration = stop - start
    print('full run takes')
    print(duration/3600)
    print('hours')

def load_task(imputepath, task_id, preprocess=True):
    
    cached_data_path = imputepath + f"/data/{task_id}_{preprocess}.pkl"
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
            if not os.path.exists(imputepath + f"/data/"):
                os.makedirs(imputepath + f"/data/")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    main()
    print("DONE")

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
    #print('loc2')
    for taskid in task_id_lists:
        for exp in experiments:
            #print('loc4')
            levelstr = str(level)
            save_folder = f"{base_save_folder}/{taskid}/{exp['exp_name']}_{type}_{levelstr}"
            checkpoint_folder = f"{base_save_folder}/checkpoint/{taskid}/{exp['exp_name']}_{type}_{levelstr}"
            #print('loc5')
            time.sleep(random.random()*5)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            #print('loc6')
            time.sleep(random.random()*5)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)

            print("working on ")
            print(save_folder)

            start = time.time()
            time.sleep(random.random()*5)
            duration = time.time() - start
            print(duration)

            try: 
                print("loading data")
                X_train, y_train, X_test, y_test = load_task(base_save_folder=base_save_folder, exp=exp, type=type, levelstr=levelstr, task_id=taskid, preprocess=True)
                print(y_train)
                X_train_pandas = pd.DataFrame(X_train)
                print(X_train_pandas)
                X_test_pandas = pd.DataFrame(X_test)
                print(X_test_pandas)
                X_train_missing_p, mask_train = add_missing(X_train_pandas, add_missing=level, missing_type=type)
                X_test_missing_p, mask_test = add_missing(X_test_pandas, add_missing=level, missing_type=type)
                X_train_missing_n = X_train_missing_p.to_numpy()
                X_test_missing_n = X_test_missing_p.to_numpy()

                print("running experiment 1/3 - Does large hyperparameter space improve reconstruction accuracy over simple")
                
                #Simple Impute 
                all_scores = {}
            
                if exp['exp_name'] == 'tpot2_base_normal':
                    SimpleImputeSpace = autoimpute.AutoImputer(missing_type=type, model_names=['SimpleImputer'], n_jobs=48, show_progress=False, random_state=num_runs)
                    SimpleImputeSpace.fit(X_train_missing_p)
                    print('simple fit')
                    simple_impute = SimpleImputeSpace.transform(X_test_missing_p)
                    print('simple transform')
                    print(simple_impute)
                    simple_rmse = SimpleImputeSpace.study.best_trial.value
                    simple_space = SimpleImputeSpace.study.best_trial.params
                    simple_impute = simple_impute.to_numpy()
                    print(simple_rmse)
                    print(simple_space)
                    all_scores['impute_rmse'] = simple_rmse
                    all_scores['impute_space'] = simple_space
                    imputed = simple_impute
                else:
                    #Auto Impute 
                    AutoImputeSpace = autoimpute.AutoImputer(missing_type=type, model_names=['SimpleImputer', 'IterativeImputer', 'KNNImputer', 'GAIN', 'RandomForestImputer'], n_jobs=48, show_progress=False, random_state=num_runs)
                    AutoImputeSpace.fit(X_train_missing_p)
                    print('auto fit')
                    auto_impute = AutoImputeSpace.transform(X_test_missing_p)
                    print('auto transform')
                    print(auto_impute)
                    auto_rmse = AutoImputeSpace.study.best_trial.value
                    auto_space = AutoImputeSpace.study.best_trial.params
                    auto_impute = auto_impute.to_numpy()
                    print(auto_rmse)
                    print(auto_space)
                    all_scores['impute_rmse'] = auto_rmse
                    all_scores['impute_space'] = auto_space
                    imputed = auto_impute
                
                print("running experiment 2/3 - Does reconstruction give good automl predictions")
                #this section trains off of original train data, and then tests on the original, the simpleimputed,
                #  and the autoimpute test data. This section uses the normal params since it is checking just for predictive preformance, 
                # not the role of various imputers in the tpot optimization space. 

                exp['params']['cv'] = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=num_runs)
                exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
                est = exp['automl'](**normal_params)

                print('Start est fit')
                start = time.time()
                est.fit(X_train, y_train)
                stop = time.time()
                duration = stop - start
                print('Fitted')
                if exp['automl'] is tpot.TPOTClassifier:
                    est.classes_ = est.fitted_pipeline_.classes_
                print(est.fitted_pipeline_)
                print('score start')
                train_score = score(est, X_train, y_train)
                print('train score:', train_score)
                ori_test_score = score(est, X_test, y_test)
                print('original test score:', ori_test_score)
                imputed_test_score = score(est, imputed, y_test)
                print('imputed test score:', imputed_test_score)
                print('score end')
                train_score = {f"train_{k}": v for k, v in train_score.items()}
                all_scores['train_score'] = train_score
                all_scores['ori_test_score']=ori_test_score
                all_scores['imputed_test_score'] = imputed_test_score
                all_scores["start"] = start
                all_scores["taskid"] = taskid
                all_scores["level"] = level
                all_scores["type"] = type
                all_scores["exp_name"] = 'Imputed_Predictive_Capacity'
                all_scores["name"] = openml.datasets.get_dataset(openml.tasks.get_task(taskid).dataset_id).name
                all_scores["duration"] = duration
                all_scores["run"] = num_runs
                all_scores["fit_model"] = est.fitted_pipeline_

                if exp['automl'] is tpot2.TPOTClassifier or exp['automl'] is tpot2.TPOTEstimator or exp['automl'] is  tpot2.TPOTEstimatorSteadyState:
                    with open(f"{save_folder}/est_evaluated_individuals.pkl", "wb") as f:
                        pickle.dump(est.evaluated_individuals, f)
                        print('estimator working as intended')
                print('check intended')
                with open(f"{save_folder}/est_fitted_pipeline.pkl", "wb") as f:
                    pickle.dump(est.fitted_pipeline_, f)

                with open(f"{save_folder}/all_scores.pkl", "wb") as f:
                    pickle.dump(all_scores, f)

                print('EXP2 Finished')
            

                print("running experiment 3/3 - What is the best automl settings?")

                exp['params']['cv'] = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=num_runs)
                exp['params']['periodic_checkpoint_folder'] = checkpoint_folder
                tpot_space = exp['automl'](**exp['params'])
                print(exp['automl'])
                print('Start tpot fit')
                start = time.time()
                tpot_space.fit(X_train_missing_n, y_train)
                stop = time.time()
                duration = stop - start
                print('Fitted')
                if exp['automl'] is tpot.TPOTClassifier:
                    tpot_space.classes_ = tpot_space.fitted_pipeline_.classes_
                print(tpot_space.fitted_pipeline_)
                print('score start')
                train_score = score(tpot_space, X_train_missing_n, y_train)
                print('train score:', train_score)
                test_score = score(tpot_space, X_test_missing_n, y_test)
                print('test score:', test_score)
                print('score end')
                tpot_space_scores = {}
                train_score = {f"train_{k}": v for k, v in train_score.items()}
                
                tpot_space_scores['train_score'] = train_score
                tpot_space_scores['ori_test_score']=test_score        
                tpot_space_scores["start"] = start
                tpot_space_scores["taskid"] = taskid
                tpot_space_scores["exp_name"] = exp['exp_name']
                tpot_space_scores["name"] = openml.datasets.get_dataset(openml.tasks.get_task(taskid).dataset_id).name
                tpot_space_scores["duration"] = duration
                tpot_space_scores["run"] = num_runs
                tpot_space_scores["fit_model"] = tpot_space.fitted_pipeline_

                if exp['automl'] is tpot2.TPOTClassifier or exp['automl'] is tpot2.TPOTEstimator or exp['automl'] is  tpot2.TPOTEstimatorSteadyState:
                    with open(f"{save_folder}/tpot_space_evaluated_individuals.pkl", "wb") as f:
                        pickle.dump(tpot_space.evaluated_individuals, f)

                with open(f"{save_folder}/tpot_space_fitted_pipeline.pkl", "wb") as f:
                    pickle.dump(tpot_space.fitted_pipeline_, f)

                with open(f"{save_folder}/tpot_space_scores.pkl", "wb") as f:
                    pickle.dump(tpot_space_scores, f)
                
                #return
                
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
    return

for taskid in ['2306', '2309', '2288', '2289', '2307', '359935','7320', '7323', '233211', '359938', '317615']:
    fileoutput = '/common/ketrong/AutoImputeExp/tpot2_imputetest/Impute_Experiments/regression_data/'
    csvout = pd.DataFrame(columns=['Exp1ImputeRMSE','Exp2ImputeModel','Exp2train_explained_var','Exp2train_r2', 
                                   'Exp2train_rmse', 'Exp2test_explained_var', 'Exp2test_r2', 
                                   'Exp2test_rmse', 'Exp2impute_explained_var', 'Exp2impute_r2', 
                                   'Exp2impute_rmse', 'Exp2impute_pipe', 'Exp2duration', 
                                   'Exp3ImputeModel', 'Exp3train_explained_var', 'Exp3train_r2', 'Exp3train_rmse', 
                                   'Exp3impute_explained_var', 'Exp3impute_r2', 'Exp3impute_rmse', 
                                   'Exp3impute_pipe', 'Exp3duration'], 
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
                normalpath = '/common/ketrong/AutoImputeExp/tpot2_imputetest/Impute_Experiments/logs/'+ taskid + exp + item + lvl
                imputepath = '/common/ketrong/AutoImputeExp/tpot2_imputetest/Impute_Experiments/logs/'+ taskid + exp + item + lvl
                
                try:
                    with open(normalpath + 'all_scores.pkl', 'rb') as file:
                        my_object = pickle.load(file)
                        #print(my_object)
                    with open(normalpath + 'est_fitted_pipeline.pkl', 'rb') as file:
                        my_object_pipeline = pickle.load(file)
                    with open(imputepath + 'tpot_space_scores.pkl', 'rb') as file:
                        my_run = pickle.load(file)
                        #print(my_run)
                    with open(imputepath + 'tpot_space_fitted_pipeline.pkl', 'rb') as file:
                        my_run_pipeline = pickle.load(file)
                    csvout.loc[exp+item+lvl] = pd.Series({'Exp1ImputeRMSE': my_object['impute_rmse'] ,'Exp2ImputeModel': str(my_object['impute_space']),'Exp2train_explained_var': my_object['train_score']['train_explained_var'],'Exp2train_r2': my_object['train_score']['train_r2'], 
                                        'Exp2train_rmse': my_object['train_score']['train_rmse'], 'Exp2test_explained_var': my_object['ori_test_score']['explained_var'], 'Exp2test_r2': my_object['ori_test_score']['r2'], 
                                        'Exp2test_rmse': my_object['ori_test_score']['rmse'], 'Exp2impute_explained_var': my_object['imputed_test_score']['explained_var'], 'Exp2impute_r2': my_object['imputed_test_score']['r2'], 
                                        'Exp2impute_rmse': my_object['imputed_test_score']['rmse'], 'Exp2impute_pipe': my_object_pipeline, 'Exp2duration': my_object['duration'], 
                                        'Exp3ImputeModel': my_run_pipeline, 'Exp3train_explained_var': my_run['train_score']['train_explained_var'], 'Exp3train_r2': my_run['train_score']['train_r2'], 'Exp3train_rmse': my_run['train_score']['train_rmse'], 
                                        'Exp3impute_explained_var': my_run['ori_test_score']['explained_var'], 'Exp3impute_r2': my_run['ori_test_score']['r2'], 'Exp3impute_rmse': my_run['ori_test_score']['rmse'], 
                                        'Exp3impute_pipe': my_run_pipeline, 'Exp3duration': my_run['duration']})
                    
                except:
                    print(taskid+item+lvl+' failed')
    output = csvout.to_csv(fileoutput+taskid+'.csv')

