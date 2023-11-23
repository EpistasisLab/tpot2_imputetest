import tpot2
import numpy as np
import sklearn.metrics
import sklearn
import argparse
import utils
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
                    'max_size' : np.inf,
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
    
    experiments = [
            {
            'automl': tpot2.TPOTEstimator,
            'exp_name' : 'tpot2_base_normal',
            'params': simple_and_normal_params,
            },
            {
            'automl': tpot2.TPOTEstimator,
            'exp_name' : 'tpot2_base_imputation',
            'params': imputation_params_and_normal_params,
            },
            ]
    #try with 67 / 69 benchmark sets
    '''
    task_id_lists = [
                    6, 26, 30, 32, 137, 151, 183, 184, 189, 197, 198, 215, 216, 218, 
                    251, 287, 310, 375, 725, 728, 737, 803, 823, 847, 871, 881, 901,
                    923, 1046, 1120, 1193, 1199, 1200, 1213, 1220, 1459, 1471, 1481,
                    1489, 1496, 1507, 1526, 1558, 4135, 4552, 23395, 23515, 23517,
                    40497, 40498, 40677, 40685, 40701, 40922, 40983, 41027, 41146,
                    41671, 42183, 42192, 42225, 42477, 42493, 42545, 42636, 42688,
                    42712]
    '''
    task_id_lists = [6]
                    
    utils.loop_through_tasks(experiments, task_id_lists, base_save_folder, num_runs)



if __name__ == '__main__':
    main()
    print("DONE")