import tpot2
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from .transformers import RandomForestImputer, GAINImputer
from .param_grids import params_SimpleImpute, params_IterativeImpute, params_KNNImpute, params_RandomForestImpute, params_GAINImpute

imputation_config_dict = {
                SimpleImputer: params_SimpleImpute, 
                IterativeImputer: params_IterativeImpute,
                KNNImputer: params_KNNImpute,
                RandomForestImputer: params_RandomForestImpute,
                GAINImputer: params_GAINImpute
}

imputation_params =  {
               'root_config_dict':imputation_config_dict,
               'leaf_config_dict': None,
               'inner_config_dict':None,
               'max_size' : 1,
               'linear_pipeline' : True
               }

normal_params =  {
                'root_config_dict':["classifiers"],
                'leaf_config_dict': None,
                'inner_config_dict': ["selectors", "transformers"],
                'max_size' : np.max,
                'linear_pipeline' : True
                }

imputation_params_and_normal_params = {
               'root_config_dict': {"Recursive" : normal_params},
               'leaf_config_dict': {"Recursive" : imputation_params},
               'inner_config_dict': None,
               'max_size' : np.max,
               'linear_pipeline' : True
               }

