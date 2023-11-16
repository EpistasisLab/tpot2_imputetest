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
                'linear_pipeline' : True,

                'scorers':['neg_log_loss', tpot2.objectives.complexity_scorer],
                'scorers_weights':[1,-1],
                'other_objective_functions':[],
                'other_objective_functions_weights':[],
                
                'population_size' : n_jobs,
                'survival_percentage':1, 
                'initial_population_size' : n_jobs,
                'generations' : None, 
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

imputation_params_and_normal_params = {
                'root_config_dict': {"Recursive" : normal_params},
                'leaf_config_dict': {"Recursive" : imputation_params},
                'inner_config_dict': None,
                'max_size' : np.max,
                'linear_pipeline' : True,

                'scorers':['neg_log_loss', tpot2.objectives.complexity_scorer],
                'scorers_weights':[1,-1],
                'other_objective_functions':[],
                'other_objective_functions_weights':[],
                
                'population_size' : n_jobs,
                'survival_percentage':1, 
                'initial_population_size' : n_jobs,
                'generations' : None, 
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

