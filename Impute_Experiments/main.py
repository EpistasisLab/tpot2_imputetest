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
                    'max_size' : 1,
                    'linear_pipeline' : True,
                    }

    imputation_params_and_normal_params = {
                    'root_config_dict': {"Recursive" : normal_params},
                    'leaf_config_dict': {"Recursive" : imputation_params},
                    'inner_config_dict': None,
                    'max_size' : 1,
                    'linear_pipeline' : True,

                    'scorers':['neg_log_loss', tpot2.objectives.complexity_scorer],
                    'scorers_weights':[1,-1],
                    'other_objective_functions':[],
                    'other_objective_functions_weights':[],
                    
                    'population_size' : n_jobs,
                    'survival_percentage':1, 
                    'initial_population_size' : n_jobs,
                    'generations' : 50, 
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
                    'max_size' : 1,
                    'linear_pipeline' : True,

                    'scorers':['neg_log_loss', tpot2.objectives.complexity_scorer],
                    'scorers_weights':[1,-1],
                    'other_objective_functions':[],
                    'other_objective_functions_weights':[],
                    
                    'population_size' : n_jobs,
                    'survival_percentage':1, 
                    'initial_population_size' : n_jobs,
                    'generations' : 50, 
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
    ''' removed:{   6, 26, 30, 32, 215, 218, 251, 4552,
                    137, 151, 183, 184, 189, 197, 198, 216, 
                    287, 310, 375, 725, 728, 737, 803, 823, 847, 871, 881, 901,
                    923, 1046, 1120, 1193, 1199, 1200, 1213, 1220, 1459, 1471, 1481,
                    1489, 1496, 1507, 1526, 1558, 4135, 23395, 23515, 23517,
                    40497, 40498, 40677, 40685, 40701, 40922, 40983, 41027, 41146,
                    41671, 42183, 42192, 42225, 42477, 42493, 42545, 42636, 42688,
                    42712}
    completed : 6, 26, 30, 32, 2142, 14953, 206, 219,2075, 2076, 2280,
                    3483, 3510, 3591, 3594, 3603, 3668, 3688, 3712, 3735, 3745, 
                     3764, 3786, 3899, 3954, 7295, 14964, 9983, 9972, 9952, 9959, 9943,
                     9942, 9899, 34539, 167120, 145943, 145681, '146204', '146212', '167141', '167212', '146820', '167119', 189865, 189773
    new_task_list_class = [
                    6, 26, 30, 32, 2142, 14953, 206, 219, 2075, 2076, 2280,
                    3483, 3510, 3591, 3594, 3603, 3668, 3688, 3712, 3735, 3745, 
                    3764, 3786, 3899, 3954, 7295, 14964, 9983, 9972, 9952, 9959, 9943, 9942, 
                    9899, 34539, 167120, 145943, 145681, 146204, 146212, 167141, 167212, 146820, 
                    167119, 189865, 189773, ]
    Ran: 6, 26, 30, 32, 2142, 14953, 206, 219 

    215, 218, 197, 216, 287, 1193, 1199, 42225, 42688, 42712,

    Completed: 2306, 2309, 2288, 2289, 2307, 359935, , 7320, 7323,
                                233211, 359938, 317615
    new_task_list_regression = [2306, 2309, 2288, 2289, 2307, 359935, 7320, 7323,
                                233211, 359938, 317615

                            ]
    1200, 1213, 23395, 23515, 42183, 42192, 42477, 42493, 42636, 
    clustering? = [127098, 127111, 128649, 128681, 295876, 295886, 296301, 296308,
                    296351, 296476,  ]
    
    additional_regression = [359946, 4774, 359952, 4769, 7393, 360969, 190419, 233169, 360966, 317615]
317615
    Top priority reg: [190419, 317615], [26, 2075, 3380, 14953, 32, 34539, 145681]
    Top priority class:
    

    task_id_lists = [
                    6, 26, 30, 32, 215, 218, 251, 4552
                    ]
                218, 251, 4552 wrong, need new task list for all.  isnt the right data set. Need to process these from the dataset pull. 
    '''
    task_id_lists = [3764]
    
    print('starting loops')
    start = time.time()
    utils.loop_through_tasks(experiments, task_id_lists, base_save_folder, num_runs)
    stop = time.time()
    duration = stop - start
    print('full run takes')
    print(duration/3600)
    print('hours')


if __name__ == '__main__':
    main()
    print("DONE")