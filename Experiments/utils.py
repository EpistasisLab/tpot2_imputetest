import openml
import tpot2
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, \
                             roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             f1_score)
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

def score(est, X, y):

    try:
        this_auroc_score = sklearn.metrics.get_scorer("roc_auc_ovr")(est, X, y)
    except:
        y_preds = est.predict(X)
        y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
        this_auroc_score = roc_auc_score(y, y_preds_onehot, multi_class="ovr")
    
    try:
        this_logloss = sklearn.metrics.get_scorer("neg_log_loss")(est, X, y)*-1
    except:
        y_preds = est.predict(X)
        y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
        this_logloss = log_loss(y, y_preds_onehot)

    this_accuracy_score = sklearn.metrics.get_scorer("accuracy")(est, X, y)
    this_balanced_accuracy_score = sklearn.metrics.get_scorer("balanced_accuracy")(est, X, y)


    return { "auroc": this_auroc_score,
            "accuracy": this_accuracy_score,
            "balanced_accuracy": this_balanced_accuracy_score,
            "logloss": this_logloss,
    }



#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id, preprocess=True):
    
    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
    print(cached_data_path)
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
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

            
            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test


def loop_through_tasks(experiments, task_id_lists, base_save_folder):
    for taskid in task_id_lists:
        for level in [1, 10, 30, 50]:
            for type in ['MCAR', 'MNAR', 'MAR']:
                for exp in experiments:
                    save_folder = f"{base_save_folder}/{exp['exp_name']}_{taskid}_{level}_{type}"
                    time.sleep(random.random()*5)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    else:
                        continue

                    print("working on ")
                    print(save_folder)

                    try: 

                        print("loading data")
                        X_train, y_train, X_test, y_test = load_task(taskid, preprocess=True)
                        

                        print("starting ml")
                        exp['params']['cv'] = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=run)
                        exp['params']['periodic_checkpoint_folder'] = f"/home/ribeirop/common/Projects/tpot_digen_paper1/tpot2_paper_1/checkpoint/{exp['exp_name']}_{taskid}_{run}"
                        est = exp['automl'](**exp['params'])

                        start = time.time()
                        est.fit(X_train, y_train)
                        duration = time.time() - start
                        
                        if type(est) is tpot.TPOTClassifier:
                            est.classes_ = est.fitted_pipeline_.classes_

                        train_score = score(est, X_train, y_train)
                        test_score = score(est, X_test, y_test)

                        all_scores = {}
                        train_score = {f"train_{k}": v for k, v in train_score.items()}
                        all_scores.update(train_score)
                        all_scores.update(test_score)

                        
                        all_scores["start"] = start
                        all_scores["taskid"] = taskid
                        all_scores["exp_name"] = exp['exp_name']
                        #all_scores["name"] = openml.datasets.get_dataset(openml.tasks.get_task(taskid).dataset_id).name
                        all_scores["duration"] = duration
                        all_scores["run"] = run

                        if type(est) is tpot2.TPOTClassifier or type(est) is tpot2.TPOTEstimator or type(est) is  tpot2.TPOTEstimatorSteadyState:
                            with open(f"{save_folder}/evaluated_individuals.pkl", "wb") as f:
                                pickle.dump(est.evaluated_individuals, f)

                        
                        with open(f"{save_folder}/fitted_pipeline.pkl", "wb") as f:
                            pickle.dump(est.fitted_pipeline_, f)


                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(all_scores, f)

                        return
                    except Exception as e:
                        trace =  traceback.format_exc() 
                        pipeline_failure_dict = {"taskid": taskid, "exp_name": exp['exp_name'], "run": run, "error": str(e), "trace": trace}
                        print("failed on ")
                        print(save_folder)
                        print(e)
                        print(trace)

                        with open(f"{save_folder}/failed.pkl", "wb") as f:
                            pickle.dump(pipeline_failure_dict, f)

                        return
    
    print("all finished")