import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import time
import utils
import autoutils
import warnings
warnings.filterwarnings("ignore")


class AutoImputer():
  def __init__(self, fname = None, internal_folds=10, n_trials=200, random_state = 42, CV_state = True, added_missing = 0.05, missing_type: str = 'MAR', model_names: list = ['SimpleImputer' , 'IterativeImputer','KNNImputer', 'GAIN', 'RandomForestImputer'], sampler = optuna.samplers.TPESampler(), direction ='minimize', n_jobs = 1, show_progress = True, garbage_collect=True):
    '''
    Loads in meta-parameters. 

    fname = str, contains relative path to csv file containing missingness set. 
    Combine after running missing_function to also get mask version of the missing set. Needed for GAIN and VAE. 

    loss_function: function : set category to optimize for. options: cv-rmse, rmse

    meta_cv_folds: cross validation outside of the optimization function

    sampler_type: optuna optimization algorithm

    n_trials: number of trials in the parameter space

    random_state: sets reproducable sampling. 
    '''
    self.fname = fname
    self.n_trials=n_trials
    self.random_state = random_state
    self.internal_folds = internal_folds
    self.acc_model = sklearn.linear_model.LogisticRegression()
    self.CV_state = CV_state
    self.added_missing = added_missing
    self.missing_type = missing_type
    self.model_names = model_names
    self.sampler = sampler
    self.direction = direction
    self.n_jobs = n_jobs
    self.show_progress = show_progress
    self.garbage_collect = garbage_collect

  def fit(self, X: pd.DataFrame, y: pd.DataFrame=None):
    start_time = time.time()
    self.missing_set_train, self.masked_set_train = utils.add_missing(X, add_missing=self.added_missing, missing_type=self.missing_type)
    if self.CV_state == False:
      splitting = ShuffleSplit(n_splits=1, random_state=self.random_state)
    else:
      splitting = KFold(n_splits=self.internal_folds, random_state=self.random_state, shuffle=True)
    def obj(trial):
      my_params = autoutils.trial_suggestion(trial, self.model_names, column_len=len(X.columns), random_state=self.random_state)
      trial.set_user_attr("out_params", my_params)
      my_model = autoutils.MyModel(random_state = self.random_state, **my_params)
      return autoutils.score(trial, splitting, my_model, X, self.missing_set_train, self.masked_set_train)
    self.study = optuna.create_study(sampler=self.sampler, direction=self.direction)
    self.study.optimize(obj, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=self.show_progress, gc_after_trial=self.garbage_collect)
    self.best_model = autoutils.MyModel(random_state = self.random_state, **self.study.best_trial.user_attrs["out_params"]) 
    self.best_model.fit(X)
    stop_time = time.time()
    self.fit_time = stop_time - start_time
    
  def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
    transstart_time = time.time()
    #self.missing_set_test, self.masked_set_test = utils.add_missing(X, add_missing=self.added_missing, missing_type=self.missing_type)
    imputed = self.best_model.transform(X)
    if 'numpy' in str(type(imputed)):
      imputed = pd.DataFrame(imputed, columns=X.columns.values)
    transstop_time = time.time()
    self.transform_time = transstop_time - transstart_time
    return imputed

  def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
    self.fit(X, y)
    imputed_set = self.transform(X, y)
    return imputed_set
  
  def missing_sets(self):
    return self.missing_set_train, self.masked_set_train, self.missing_set_test, self.masked_set_test