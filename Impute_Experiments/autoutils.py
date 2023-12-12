import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from param_grids import params_SimpleImpute, params_IterativeImpute, params_KNNImpute, params_RandomForestImpute, params_GAINImpute
from transformers import RandomForestImputer, GAINImputer


def trial_suggestion(trial: optuna.trial.Trial, model_names,column_len,random_state):
    model_name = trial.suggest_categorical('model_name', model_names)# Model names set up to run on multiple or individual models. Options include: 'SimpleImputer' , 'IterativeImputer','KNNImputer', 'VAEImputer', 'GAIN', 'Opt.SVM', 'Opt.Trees'.  Not yet working: 'DLImputer', Random Forest Imputer 
    my_params = {}
    match model_name:
      case 'SimpleImputer':
        my_params = params_SimpleImpute(trial, model_name)  #Takes data from each column to input a value for all missing data. 
      case 'IterativeImputer':
        my_params = params_IterativeImpute(trial,model_name) #Uses the dependence between columns in the data set to predict for the one column. Predictions occur through a variety of strategies.
      case 'KNNImputer':
        my_params = params_KNNImpute(trial,model_name) #uses nearest neighbors to predict missing values with k-neighbors in n-dimensional space with known values.
      case 'GAIN':
        my_params = params_GAINImpute(trial,model_name) #Uses a generative adversarial network model to predict values. 
      case 'RandomForestImputer': #uses a hyperparameter optimized SVM model to predict values to impute.
        my_params = params_RandomForestImpute(trial, model_name)
    my_params['model_name'] = model_name
    return my_params
  
def MyModel(random_state, **params):
    these_params = params
    model_name = these_params['model_name']
    del these_params['model_name']

    match model_name:
        case 'SimpleImputer':
            this_model = SimpleImputer(
                **these_params
                )
        case 'IterativeImputer':
            this_model = IterativeImputer(
                **these_params
                )
        case 'KNNImputer':
            this_model = KNNImputer( 
                **these_params
                )
        case 'GAIN':
            this_model = GAINImputer(
                **these_params
                )
        case 'RandomForestImputer': #uses a hyperparameter optimized SVM model to predict values to impute.
            this_model = RandomForestImputer(
                **these_params
            )
    return this_model

def score(trial: optuna.trial.Trial, splitting, my_model, X: pd.DataFrame, missing_set: pd.DataFrame, masked_set:pd.DataFrame):
    avg_cv_rmse = []
    for i, (train_index, test_index) in enumerate(splitting.split(X)):
        missing_train, missing_test, X_test, masked_test = missing_set.iloc[train_index], missing_set.iloc[test_index], X.iloc[test_index], masked_set.iloc[test_index]
        try: 
            my_model.fit(missing_train)
            imputed = my_model.transform(missing_test)

            if 'numpy' in str(type(imputed)):
                imputed = pd.DataFrame(imputed, columns=missing_set.columns.values)

            if imputed.isnull().sum().sum() > 0: 
                rmse_val = np.inf
                return rmse_val
            
            rmse_val = rmse_loss(ori_data=X_test.to_numpy(), imputed_data=imputed.to_numpy(), data_m=np.multiply(masked_test.to_numpy(),1))
            avg_cv_rmse.append(rmse_val)
        except:
            rmse_val = np.inf
            return rmse_val
    cv_rmse = sum(avg_cv_rmse)/float(len(avg_cv_rmse))
    trial.set_user_attr('cv-rmse', cv_rmse)
    return cv_rmse

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