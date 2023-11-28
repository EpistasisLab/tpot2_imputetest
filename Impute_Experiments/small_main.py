import tpot2
import numpy as np
import sklearn.metrics
import sklearn
import argparse
import utils
import time
import random
import autoimpute
import os
import transformers
import autoutils
import pandas as pd
import sklearn.datasets
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

    save_folder = f"{base_save_folder}/small_run"
    checkpoint_folder = f"{base_save_folder}/checkpoint/small_run"
    time.sleep(random.random()*5)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    time.sleep(random.random()*5)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    print("working on ")
    print(save_folder)

    print("loading data")
    X_train, y_train, X_test, y_test = utils.load_task(26, preprocess=True)
    X_train_pandas = pd.DataFrame(X_train)
    X_test_pandas = pd.DataFrame(X_test)

    print("starting impute modules")
    X_train_missing, X_train_mask = utils.add_missing(X_train_pandas, add_missing=0.5, missing_type='MNAR')
    X_test_missing, X_test_mask = utils.add_missing(X_test_pandas, add_missing=0.5, missing_type='MNAR')
    X_train_missing = X_train_missing.to_numpy()
    X_test_missing = X_test_missing.to_numpy()
    X_train_mask = X_train_mask.to_numpy()
    X_test_mask = X_test_mask.to_numpy()


    print("running experiment 1/3 - Does large hyperparameter space improve reconstruction accuracy over simple")

    gainmodel = GAINImputer(batch_size=128, hint_rate=0.9, alpha=100, iterations=10000)
    gainmodel.fit(X=X_train_missing)
    gain_test_missing = gainmodel.transform(X=X_test_missing)

    gain_rmse = autoutils.rmse_loss(ori_data=X_test, imputed_data=gain_test_missing, data_m=X_test_mask)
    print(gain_rmse)

    randmodel = RandomForestImputer()
    randmodel.fit(X=X_train_missing)
    rand_test_missing = randmodel.transform(X=X_test_missing)

    rand_rmse = autoutils.rmse_loss(ori_data=X_test, imputed_data=rand_test_missing, data_m=X_test_mask)
    print(rand_rmse)

if __name__ == '__small_main__':
    main()
    print("DONE")