import numpy as np
import numbers
import warnings
from collections import Counter

import numpy as np
import numpy.ma as ma
from scipy import sparse as sp
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import _MissingValues, StrOptions
from sklearn.impute._base import _BaseImputer, _check_inputs_dtype, _most_frequent
from sklearn.utils import _is_pandas_na, is_scalar_nan
from contextlib import suppress
from sklearn.utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
from sklearn.utils._mask import _get_mask
from sklearn.utils.sparsefuncs import _get_median
from sklearn.metrics import pairwise_distances_chunked

class RandomForestImputer(_BaseImputer):
    """Imputer for completing missing values with random forest strategies.

    Replace missing values using sklearn's built in random forest regressor function along each column.

    Written by Gabriel Ketron Nov 16th, 2023
    Contact: Gabriel.Ketron@cshs.org

    Parameters
    ----------
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.

    strategy : str, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

    fill_value : str or numerical value, default=None
        When strategy == "constant", `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0` except when `strategy="constant"`
        in which case `fill_value` will be used instead.

        .. versionadded:: 1.2

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.
        Computing statistics can result in `np.nan` values.
        During :meth:`transform`, features corresponding to `np.nan`
        statistics will be discarded.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    IterativeImputer : Multivariate imputer that estimates values to impute for
        each feature with missing values from all the others.
    KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples.

    Notes
    -----
    Columns which only contained missing values at :meth:`fit` are discarded
    upon :meth:`transform` if strategy is not `"constant"`.

    In a prediction context, simple imputation usually performs poorly when
    associated with a weak learner. However, with a powerful learner, it can
    lead to as good or better performance than complex imputation such as
    :class:`~sklearn.impute.IterativeImputer` or :class:`~sklearn.impute.KNNImputer`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    SimpleImputer()
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    For a more detailed example see
    :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.
    """
    # Add space objects here

    _parameter_constraints: dict = {
        **_BaseImputer._parameter_constraints,
        "strategy": [StrOptions({"mean", "median", "most_frequent", "constant"})],
        "fill_value": "no_validation",  # any object is valid
        "copy": ["boolean"],
    }

    def __init__(
        self,
        *,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
    ):
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        """Helper function to impute a single column.

        Parameters
        ----------
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.

        n_neighbors : int
            Number of neighbors to consider.

        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.

        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.

        Returns
        -------
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
        # Get donors
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # Get weight matrix from distance matrix
        donors_dist = dist_pot_donors[
            np.arange(donors_idx.shape[0])[:, None], donors_idx
        ]

        weight_matrix = _get_weights(donors_dist, self.weights)

        # fill nans with zeros
        if weight_matrix is not None:
            weight_matrix[np.isnan(weight_matrix)] = 0.0

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)

        return np.ma.average(donors, axis=1, weights=weight_matrix).data
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the imputer on X. 
        
        Parameters
        -----------
        X : array-like shape of (n_samples, n_features)
            Input data, where 'n_samples' is the number of samples and
            'n_features is the number of features. 
        
        y: Ignored
            Not used, present here for API consitency by convention. 
        
        Returns
        -------
        self: : object
            The fitted 'RandomForestImputer' class instance.
        """
        # Check data integrity and calling arguments
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=force_all_finite,
            copy=self.copy,
        )

        self._fit_X = X
        self._mask_fit_X = _get_mask(self._fit_X, self.missing_values)
        self._valid_mask = ~np.all(self._mask_fit_X, axis=0)

        super()._fit_indicator(self._mask_fit_X)

        return self


    def transform(self, X):
        """Impute all missing values in `X`.

        Parameters
        ----------
        X : array-like shape of (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        X : array-like of shape (n_samples, n_output_features)
            The imputed dataset. `n_output_features` is the number of features
            that is not always missing during `fit`.
        """
        
        check_is_fitted(self)
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=force_all_finite,
            copy=self.copy,
            reset=False,
        )

        mask = _get_mask(X, self.missing_values)
        mask_fit_X = self._mask_fit_X
        valid_mask = self._valid_mask

        X_indicator = super()._transform_indicator(mask)

        # Removes columns where the training data is all nan
        if not np.any(mask):
            # No missing values in X
            if self.keep_empty_features:
                Xc = X
                Xc[:, ~valid_mask] = 0
            else:
                Xc = X[:, valid_mask]

            # Even if there are no missing values in X, we still concatenate Xc
            # with the missing value indicator matrix, X_indicator.
            # This is to ensure that the output maintains consistency in terms
            # of columns, regardless of whether missing values exist in X or not.
            return super()._concatenate_indicator(Xc, X_indicator)

        row_missing_idx = np.flatnonzero(mask.any(axis=1))

        non_missing_fix_X = np.logical_not(mask_fit_X)

        # Maps from indices from X to indices in dist matrix
        dist_idx_map = np.zeros(X.shape[0], dtype=int)
        dist_idx_map[row_missing_idx] = np.arange(row_missing_idx.shape[0])

        def process_chunk(dist_chunk, start):
            row_missing_chunk = row_missing_idx[start : start + len(dist_chunk)]

            # Find and impute missing by column
            for col in range(X.shape[1]):
                if not valid_mask[col]:
                    # column was all missing during training
                    continue

                col_mask = mask[row_missing_chunk, col]
                if not np.any(col_mask):
                    # column has no missing values
                    continue

                (potential_donors_idx,) = np.nonzero(non_missing_fix_X[:, col])

                # receivers_idx are indices in X
                receivers_idx = row_missing_chunk[np.flatnonzero(col_mask)]

                # distances for samples that needed imputation for column
                dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][
                    :, potential_donors_idx
                ]

                # receivers with all nan distances impute with mean
                all_nan_dist_mask = np.isnan(dist_subset).all(axis=1)
                all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

                if all_nan_receivers_idx.size:
                    col_mean = np.ma.array(
                        self._fit_X[:, col], mask=mask_fit_X[:, col]
                    ).mean()
                    X[all_nan_receivers_idx, col] = col_mean

                    if len(all_nan_receivers_idx) == len(receivers_idx):
                        # all receivers imputed with mean
                        continue

                    # receivers with at least one defined distance
                    receivers_idx = receivers_idx[~all_nan_dist_mask]
                    dist_subset = dist_chunk[dist_idx_map[receivers_idx] - start][
                        :, potential_donors_idx
                    ]

                n_neighbors = min(self.n_neighbors, len(potential_donors_idx))
                value = self._calc_impute(
                    dist_subset,
                    n_neighbors,
                    self._fit_X[potential_donors_idx, col],
                    mask_fit_X[potential_donors_idx, col],
                )
                X[receivers_idx, col] = value

        # process in fixed-memory chunks
        gen = pairwise_distances_chunked(
            X[row_missing_idx, :],
            self._fit_X,
            metric=self.metric,
            missing_values=self.missing_values,
            force_all_finite=force_all_finite,
            reduce_func=process_chunk,
        )
        for chunk in gen:
            # process_chunk modifies X in place. No return value.
            pass

        if self.keep_empty_features:
            Xc = X
            Xc[:, ~valid_mask] = 0
        else:
            Xc = X[:, valid_mask]

        return super()._concatenate_indicator(Xc, X_indicator)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(self, input_features)
        non_missing_mask = np.logical_not(_get_mask(self.statistics_, np.nan))
        names = input_features[non_missing_mask]
        return self._concatenate_indicator_feature_names_out(names, input_features)
