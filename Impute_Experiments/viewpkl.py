import tpot2
import dill as pickle
import pandas as pd

for taskid in ['26', '2280', '190419', '233211', '360966']:
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
                    with open(imputepath + 'tpot_space_evaluated_individuals.pkl', 'rb') as file:
                        my_run_pipeline = pickle.load(file)
                        print(my_run_pipeline)
                    csvout.loc[exp+item+lvl] = pd.Series({'Exp1ImputeRMSE': my_object['impute_rmse'] ,'Exp2ImputeModel': str(my_object['impute_space']),'Exp2train_explained_var': my_object['train_score']['train_explained_var'],'Exp2train_r2': my_object['train_score']['train_r2'], 
                                        'Exp2train_rmse': my_object['train_score']['train_rmse'], 'Exp2test_explained_var': my_object['ori_test_score']['explained_var'], 'Exp2test_r2': my_object['ori_test_score']['r2'], 
                                        'Exp2test_rmse': my_object['ori_test_score']['rmse'], 'Exp2impute_explained_var': my_object['imputed_test_score']['explained_var'], 'Exp2impute_r2': my_object['imputed_test_score']['r2'], 
                                        'Exp2impute_rmse': my_object['imputed_test_score']['rmse'], 'Exp2impute_pipe': my_object_pipeline, 'Exp2duration': my_object['duration'], 
                                        'Exp3ImputeModel': my_run_pipeline, 'Exp3train_explained_var': my_run['train_score']['train_explained_var'], 'Exp3train_r2': my_run['train_score']['train_r2'], 'Exp3train_rmse': my_run['train_score']['train_rmse'], 
                                        'Exp3impute_explained_var': my_run['ori_test_score']['explained_var'], 'Exp3impute_r2': my_run['ori_test_score']['r2'], 'Exp3impute_rmse': my_run['ori_test_score']['rmse'], 
                                        'Exp3impute_pipe': my_run_pipeline, 'Exp3duration': my_run['duration']})
                    print(taskid+item+lvl+' passed')

                except:
                    print(taskid+item+lvl+' failed')
                
    output = csvout.to_csv(fileoutput+taskid+'.csv')
    print(taskid + 'complete')
