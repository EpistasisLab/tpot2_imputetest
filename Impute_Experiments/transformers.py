"""MissForest Imputer for Missing Data"""
# Author: Ashim Bhattarai
# License: GNU General Public License v3 (GPLv3)

import warnings

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from scipy.stats import mode
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


__all__ = [
    'MissForest',
]


class RandomForestImputer(BaseEstimator, TransformerMixin):
    """Missing value imputation using Random Forests.

    MissForest imputes missing values using Random Forests in an iterative
    fashion. By default, the imputer begins imputing missing values of the
    column (which is expected to be a variable) with the smallest number of
    missing values -- let's call this the candidate column.
    The first step involves filling any missing values of the remaining,
    non-candidate, columns with an initial guess, which is the column mean for
    columns representing numerical variables and the column mode for columns
    representing categorical variables. After that, the imputer fits a random
    forest model with the candidate column as the outcome variable and the
    remaining columns as the predictors over all rows where the candidate
    column values are not missing.
    After the fit, the missing rows of the candidate column are
    imputed using the prediction from the fitted Random Forest. The
    rows of the non-candidate columns act as the input data for the fitted
    model.
    Following this, the imputer moves on to the next candidate column with the
    second smallest number of missing values from among the non-candidate
    columns in the first round. The process repeats itself for each column
    with a missing value, possibly over multiple iterations or epochs for
    each column, until the stopping criterion is met.
    The stopping criterion is governed by the "difference" between the imputed
    arrays over successive iterations. For numerical variables (num_vars_),
    the difference is defined as follows:

     sum((X_new[:, num_vars_] - X_old[:, num_vars_]) ** 2) /
     sum((X_new[:, num_vars_]) ** 2)

    For categorical variables(cat_vars_), the difference is defined as follows:

    sum(X_new[:, cat_vars_] != X_old[:, cat_vars_])) / n_cat_missing

    where X_new is the newly imputed array, X_old is the array imputed in the
    previous round, n_cat_missing is the total number of categorical
    values that are missing, and the sum() is performed both across rows
    and columns. Following [1], the stopping criterion is considered to have
    been met when difference between X_new and X_old increases for the first
    time for both types of variables (if available).

    Parameters
    ----------
    NOTE: Most parameter definitions below are taken verbatim from the
    Scikit-Learn documentation at [2] and [3].

    max_iter : int, optional (default = 10)
        The maximum iterations of the imputation process. Each column with a
        missing value is imputed exactly once in a given iteration.

    decreasing : boolean, optional (default = False)
        If set to True, columns are sorted according to decreasing number of
        missing values. In other words, imputation will move from imputing
        columns with the largest number of missing values to columns with
        fewest number of missing values.

    missing_values : np.nan, integer, optional (default = np.nan)
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    criterion : tuple, optional (default = ('mse', 'gini'))
        The function to measure the quality of a split.The first element of
        the tuple is for the Random Forest Regressor (for imputing numerical
        variables) while the second element is for the Random Forest
        Classifier (for imputing categorical variables).

    n_estimators : integer, optional (default=100)
        The number of trees in the forest.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        NOTE: This parameter is only applicable for Random Forest Classifier
        objects (i.e., for categorical variables).

    Attributes
    ----------
    statistics_ : Dictionary of length two
        The first element is an array with the mean of each numerical feature
        being imputed while the second element is an array of modes of
        categorical features being imputed (if available, otherwise it
        will be None).

    References
    ----------
    * [1] Stekhoven, Daniel J., and Peter Bühlmann. "MissForest—non-parametric
      missing value imputation for mixed-type data." Bioinformatics 28.1
      (2011): 112-118.
    * [2] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.
      RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    * [3] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.
      RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

    Examples
    --------
    >>> from missingpy import MissForest
    >>> nan = float("NaN")
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = MissForest(random_state=1337)
    >>> imputer.fit_transform(X)
    Iteration: 0
    Iteration: 1
    Iteration: 2
    array([[1.  , 2. , 3.92 ],
           [3.  , 4. , 3. ],
           [2.71, 6. , 5. ],
           [8.  , 8. , 7. ]])
    """

    def __init__(self, max_iter=10, decreasing=False, missing_values=np.nan,
                 copy=True, n_estimators=100, criterion=('friedman_mse', 'gini'),
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=1.0,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                 verbose=0, warm_start=False, class_weight=None):

        self.max_iter = max_iter
        self.decreasing = decreasing
        self.missing_values = missing_values
        self.copy = copy
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

    def _miss_forest(self, Ximp, mask):
        """The missForest algorithm"""

        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)

            # Reg criterion
            reg_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[0]

            # Instantiate regression model
            rf_regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=reg_criterion,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start)

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get('col_modes')
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)

            # Classfication criterion
            clf_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[1]

            # Instantiate classification model
            rf_classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=clf_criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=self.warm_start,
                class_weight=self.class_weight)

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf
        col_index = np.arange(Ximp.shape[1])

        while (
                gamma_new < gamma_old or gamma_newcat < gamma_oldcat) and \
                self.iter_count_ < self.max_iter:

            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat
            # 5. loop
            for s in misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]

                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_vars_ is not None and s in self.cat_vars_:
                    rf_classifier.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_classifier.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis
                else:
                    rf_regressor.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_regressor.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

            # 9. Update gamma (stopping criterion)
            if self.cat_vars_ is not None:
                gamma_newcat = np.sum(
                    (Ximp[:, self.cat_vars_] != Ximp_old[:, self.cat_vars_])) / n_catmissing
            if self.num_vars_ is not None:
                gamma_new = np.sum((Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_]) ** 2) / np.sum((Ximp[:, self.num_vars_]) ** 2)

            #print("Iteration:", self.iter_count_)
            self.iter_count_ += 1

        return Ximp_old

    def fit(self, X, y=None, cat_vars=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        cat_vars : int or array of ints, optional (default = None)
            An int or an array containing column indices of categorical
            variable(s)/feature(s) present in the dataset X.
            ``None`` if there are no categorical variables in the dataset.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True

        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Check cat_vars type and convert if necessary
        if cat_vars is not None:
            if type(cat_vars) == int:
                cat_vars = [cat_vars]
            elif type(cat_vars) == list or type(cat_vars) == np.ndarray:
                if np.array(cat_vars).dtype != int:
                    raise ValueError(
                        "cat_vars needs to be either an int or an array "
                        "of ints.")
            else:
                raise ValueError("cat_vars needs to be either an int or an array "
                                 "of ints.")

        # Identify numerical variables
        num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
        num_vars = num_vars if len(num_vars) > 0 else None

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ['NaN', np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make initial guess for missing values
        col_means = np.nanmean(X[:, num_vars], axis=0) if num_vars is not None else None
        col_modes = mode(
            X[:, cat_vars], axis=0, nan_policy='omit')[0] if cat_vars is not \
                                                           None else None

        self.cat_vars_ = cat_vars
        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}

        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        """
        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_", "statistics_"])

        # Check data integrity
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True
        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (0 if self.num_vars_ is None else len(self.num_vars_)) \
            + (0 if self.cat_vars_ is None else len(self.cat_vars_))
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError("Incompatible dimension between the fitted "
                             "dataset and the one to be transformed.")

        # Check if anything is actually missing and if not return original X                             
        mask = _get_mask(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn("No missing value located; returning original "
                          "dataset.")
            return X

        # row_total_missing = mask.sum(axis=1)
        # if not np.any(row_total_missing):
        #     return X

        # Call missForest function to impute missing
        X = self._miss_forest(X, mask)

        # Return imputed dataset
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit MissForest and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, **fit_params).transform(X)
    
class GAINImputer(BaseEstimator, TransformerMixin):
    """Imputer for completing missing values with random forest strategies.

    Replace missing values using sklearn's built in random forest regressor function along each column.

    Written by Gabriel Ketron Nov 16th, 2023
    Adapted from Jinsung Yoon's gain function.
    Contact: Gabriel.Ketron@cshs.org

    Parameters
    ----------
    'batch_size': batch_size,
    'hint_rate': hint_rate,
    'alpha': alpha,
    'iterations': iterations
    Attributes
    ----------

    References
    ----------

    """
    def __init__(self, batch_size, hint_rate, alpha, iterations):
        self.batch_size = batch_size,
        self.hint_rate = hint_rate, 
        self.alpha = alpha
        self.iterations = iterations
        self.params = {  
        'batch_size': batch_size,
        'hint_rate': hint_rate,
        'alpha': alpha,
        'iterations': iterations
        }
            
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.gain(X, self.params)

    def fit_transform(self, X, y=None, **fit_params):
        return self.gain(X, self.params)
    
    
    def gain(self, data_x, gain_parameters):
        '''Impute missing values in data_x
        Args:
            - data_x: original data with missing values
            - gain_parameters: GAIN network parameters:
            - batch_size: Batch size
            - hint_rate: Hint rate
            - alpha: Hyperparameter
            - iterations: Iterations
        Returns:
            - imputed_data: imputed data
        '''
        if hasattr(data_x, 'dtypes'):
            data_x = data_x.to_numpy()
        # Define mask matrix
        data_m = 1-np.isnan(data_x)
        # System parameters
        batch_size = gain_parameters['batch_size']
        hint_rate = gain_parameters['hint_rate']
        alpha = gain_parameters['alpha']
        iterations = gain_parameters['iterations']
        # Other parameters
        no, dim = data_x.shape
        # Hidden state dimensions
        h_dim = int(dim)
        # Normalization
        norm_data, norm_parameters = self.normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)
        ## GAIN architecture   
        # Input placeholders
        # Data vector
        X = tf.placeholder(tf.float32, shape = [None, dim])
        # Mask vector 
        M = tf.placeholder(tf.float32, shape = [None, dim])
        # Hint vector
        H = tf.placeholder(tf.float32, shape = [None, dim])
        # Discriminator variables
        D_W1 = tf.Variable(self.xavier_init([dim*2, h_dim])) # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
        D_W2 = tf.Variable(self.xavier_init([h_dim, h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
        D_W3 = tf.Variable(self.xavier_init([h_dim, dim]))
        D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        #Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        G_W1 = tf.Variable(self.xavier_init([dim*2, h_dim]))  
        G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
        G_W2 = tf.Variable(self.xavier_init([h_dim, h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
        G_W3 = tf.Variable(self.xavier_init([h_dim, dim]))
        G_b3 = tf.Variable(tf.zeros(shape = [dim]))
        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
        ## GAIN functions
        # Generator
        def generator(x,m):
            # Concatenate Mask and Data
            inputs = tf.concat(values = [x, m], axis = 1) 
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
            # MinMax normalized output
            G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
            return G_prob
        # Discriminator
        def discriminator(x, h):
            # Concatenate Data and Hint
            inputs = tf.concat(values = [x, h], axis = 1) 
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)
            return D_prob
        ## GAIN structure
        # Generator
        G_sample = generator(X, M)
        # Combine with observed data
        Hat_X = X * M + G_sample * (1-M)
        # Discriminator
        D_prob = discriminator(Hat_X, H)
        ## GAIN loss
        D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                        + (1-M) * tf.log(1. - D_prob + 1e-8)) 
        G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
        MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
        D_loss = D_loss_temp
        G_loss = G_loss_temp + alpha * MSE_loss 
        ## GAIN solver
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
        ## Iterations
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # Start Iterations
        for it in range(iterations):    
            # Sample batch
            batch_idx = self.sample_batch_index(no, batch_size)
            X_mb = norm_data_x[batch_idx, :]  
            M_mb = data_m[batch_idx, :]  
            # Sample random vectors  
            Z_mb = self.uniform_sampler(0, 0.01, batch_size, dim) 
            # Sample hint vectors
            H_mb_temp = self.binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                    feed_dict = {M: M_mb, X: X_mb, H: H_mb})
            _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss],
                    feed_dict = {X: X_mb, M: M_mb, H: H_mb})      
        ## Return imputed data      
        Z_mb = self.uniform_sampler(0, 0.01, no, dim) 
        M_mb = data_m
        X_mb = norm_data_x          
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
        imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
        # Renormalization
        imputed_data = self.renormalization(imputed_data, norm_parameters)  
        # Rounding
        imputed_data = self.rounding(imputed_data, data_x)       
        return imputed_data

    def normalization(self, data, parameters=None):
        '''Normalize data in [0, 1] range.
        Args:
            - data: original data
        Returns:
            - norm_data: normalized data
            - norm_parameters: min_val, max_val for each feature for renormalization
        '''
        # Parameters
        _, dim = data.shape
        norm_data = data.copy()
        if parameters is None:
            # MixMax normalization
            min_val = np.zeros(dim)
            max_val = np.zeros(dim)
            # For each dimension
            for i in range(dim):
                min_val[i] = np.nanmin(norm_data[:,i])
                norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
                max_val[i] = np.nanmax(norm_data[:,i])
                norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
            # Return norm_parameters for renormalization
            norm_parameters = {'min_val': min_val,
                            'max_val': max_val}
        else:
            min_val = parameters['min_val']
            max_val = parameters['max_val']
            # For each dimension
            for i in range(dim):
                norm_data[:,i] = norm_data[:,i] - min_val[i]
                norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
            norm_parameters = parameters    
        return norm_data, norm_parameters

    def renormalization(self, norm_data, norm_parameters):
        '''Renormalize data from [0, 1] range to the original range.
        Args:
            - norm_data: normalized data
            - norm_parameters: min_val, max_val for each feature for renormalization
        Returns:
            - renorm_data: renormalized original data
        '''
        min_val = norm_parameters['min_val']
        max_val = norm_parameters['max_val']
        _, dim = norm_data.shape
        renorm_data = norm_data.copy()
        for i in range(dim):
            renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
            renorm_data[:,i] = renorm_data[:,i] + min_val[i]
        return renorm_data

    def rounding(self, imputed_data, data_x):
        '''Round imputed data for categorical variables.
        Args:
            - imputed_data: imputed data
            - data_x: original data with missing values 
        Returns:
            - rounded_data: rounded imputed data
        '''
        _, dim = data_x.shape
        rounded_data = imputed_data.copy()
        for i in range(dim):
            temp = data_x[~np.isnan(data_x[:, i]), i]
            # Only for the categorical variable
            if len(np.unique(temp)) < 20:
                rounded_data[:, i] = np.round(rounded_data[:, i])
        return rounded_data

    def rmse_loss(self, ori_data, imputed_data, data_m):
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
        nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
        denominator = np.sum(1-data_m)
        rmse = np.sqrt(nominator/float(denominator))
        return rmse

    def xavier_init(self, size):
        '''Xavier initialization.
        Args:
            - size: vector size
        Returns:
            - initialized random vector.
        '''
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape = size, stddev = xavier_stddev)
        
    def binary_sampler(self, p, rows, cols):
        '''Sample binary random variables.
        Args:
            - p: probability of 1
            - rows: the number of rows
            - cols: the number of columns
        Returns:
            - binary_random_matrix: generated binary random matrix.
        '''
        unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
        binary_random_matrix = 1*(unif_random_matrix < p)
        return binary_random_matrix

    def uniform_sampler(self, low, high, rows, cols):
        '''Sample uniform random variables.
        Args:
            - low: low limit
            - high: high limit
            - rows: the number of rows
            - cols: the number of columns
        Returns:
            - uniform_random_matrix: generated uniform random matrix.
        '''
        return np.random.uniform(low, high, size = [rows, cols])       

    def sample_batch_index(self, total, batch_size):
        '''Sample index of the mini-batch.
        Args:
            - total: total number of samples
            - batch_size: batch size
        Returns:
            - batch_idx: batch index
        '''
        total_idx = np.random.permutation(total)
        batch_idx = total_idx[:batch_size]
        return batch_idx