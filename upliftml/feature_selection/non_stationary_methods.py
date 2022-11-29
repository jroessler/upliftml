from datetime import datetime
from typing import Dict, Tuple, Any

import pandas as pd
import pyspark

from upliftml.feature_selection.stationary_methods import (
    DivergenceFilter,
    NetInformationValueFilter,
    PermutationUpliftRandomForestWrapper,
    UpliftCurveFilter,
)
from upliftml.feature_selection.utils import (
    discretizing,
    get_data_between_dates,
    linear_weighting,
    min_max_normalization
)


class FeatureImportanceOverTime:
    """
    Feature selection method over time

    The idea of Feature Importance over Time (FIoT) is straightforward:
    Instead of using all the data we have once to calculate the feature importance scores, we calculate multiple feature
    importance scores at different time steps and use a weighting schema (e.g., linear weighting) to compute the
    overall feature importance scores

    FIoT comes with two hyperparameters:
        `unit`: Number of days used for calculating a single feature importance
        `time_step`: Number of iterations to calculate the final feature importance

    Naturally, `time_step` depends on `unit` and the number of time steps you have. For example, let’s say that we have
    data from 01.07 - 31.08, that is, 8 weeks or 62 days. We want to make a prediction for the 01.09 and we want to
    select the top k features.

        If `unit` equals 7, `time_step` equals 8
        If `unit` equals 5, `time_step` equals 12

    Thus, time_step = days // unit$
    """

    def __init__(
        self,
        target_colname: str = "outcome",
        treatment_colname: str = "treatment",
        treatment_value: int = 1,
        control_value: int = 0,
        n_bins: int = 10,
        method: str = "divergence",
        date_col: str = "yyyy_mm_dd",
        date_format: str = "%Y-%m-%d",
        divergence_method: str = "KL",
        smooth: bool = True,
        factor_list: list = [],
        durf_dict: dict = None,
        n_samples: int = -1,
        n_repeats: int = 3,
        h2o_context: Any = None,
    ):
        """
        Initialize FeatureImportanceOverTime

        Args:
            target_colname (str, optional): The column name that contains the target
            treatment_colname (str, optional): The column name that contains the treatment indicators
            treatment_value (str or int, optional): The value in column <treatment_colname> that refers to the treatment group
            control_value (str or int, optional): The value in column <treatment_colname> that refers to the control group
            n_bins (int, optional): Number of bins to be used for bin-based uplift filter methods. -1 means using not discretization
            method (string, optional): Method which should be applied to calculate the feature importance at each time step. Take one of the following values {'divergence', 'permutation', 'niv', 'uc'}
            date_col (str, optional): The column name that contains the date. Can either be actual dates or int values (e.g. for different splits)
            date_format (str, optional): Formatter for date
            divergence_method (string, optional): Divergence-specific parameter. The divergence method to be used to rank the features. Taking one of the following values {'KL', 'ED', 'Chi'}.
            smooth (bool, optional): Divergence-specific parameter. Smooth label count by adding 1 in case certain labels do not occur naturally with a treatment. Prevents zero divisions.
            factor_list (list, optional): Permutation-specific parameter. List of categorical features
            durf_dict (dict, optional): Permutation-specific parameter. Hyperparamter dictionary for the distributed uplift random forest
            n_samples (int, optional): Permutation-specific parameter. Number of samples used for calculating the score
            n_repeats (int, optional): Permutation-specific parameter. Number of repeats for the permutation
            h2o_context (H2OContext, optional): H2O Context
        """
        self.target_colname = target_colname
        self.treatment_colname = treatment_colname
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.n_bins = n_bins
        self.method = method
        self.date_col = date_col
        self.date_format = date_format
        self.divergence_method = divergence_method
        self.smooth = smooth
        self.factor_list = factor_list
        self.durf_dict = durf_dict
        self.n_samples = n_samples
        self.n_repeats = n_repeats
        self.h2o_context = h2o_context

    def calculate_feature_importance(
        self, df: pyspark.sql.DataFrame, time_steps: int, dates: list, features: list, drop_columns_fs: list
    ) -> Tuple:
        """
        Calculate feature importance scores

        Args:
            df (pyspark.sql.DataFrame): DataFrame containing outcome, features, and experiment group
            time_steps (int): Number of time steps
            dates (list): List of dates tuples
            features (list): List of feature names, which are columns in the dataframe
            drop_columns_fs (list): List of features to drop (such as treatment or response)

        Returns:
            A tuple, containing:
            (pd.DataFrame): Dataframe containing the feature importance statistics
            (Dict): Dictionary containing for each feature the importance of its variables
        """
        # Discretizing features if their cardinality is higher than n_bins
        if self.method != "permutation" and self.n_bins != -1:
            df, features = discretizing(df, features, self.n_bins)

        feature_importance_scores = pd.DataFrame()
        features_variables_importances = {}  # type: Dict[str, float]

        if self.method == "divergence":
            divergence_filter = DivergenceFilter(
                target_colname=self.target_colname,
                treatment_colname=self.treatment_colname,
                treatment_value=self.treatment_value,
                control_value=self.control_value,
                n_bins=-1,
                method=self.divergence_method,
                smooth=self.smooth
            )
        elif self.method == "permutation":
            permutation_urf = PermutationUpliftRandomForestWrapper(
                durf_dict=self.durf_dict,
                factor_list=self.factor_list,
                target_colname=self.target_colname,
                treatment_colname=self.treatment_colname,
                n_samples=self.n_samples,
                n_repeats=self.n_repeats
            )
        elif self.method == "niv":
            niv = NetInformationValueFilter(
                target_colname=self.target_colname,
                treatment_colname=self.treatment_colname,
                treatment_value=self.treatment_value,
                control_value=self.control_value,
                n_bins=-1
            )
        elif self.method == "uc":
            uplift_curve = UpliftCurveFilter(
                target_colname=self.target_colname,
                treatment_colname=self.treatment_colname,
                treatment_value=self.treatment_value,
                control_value=self.control_value,
                n_bins=-1
            )

        for train_date in dates:
            df_at_t = get_data_between_dates(df, train_date[0], train_date[1], date_col=self.date_col)
            df_at_t = df_at_t.drop(*drop_columns_fs)
            str_date = train_date[1]

            if isinstance(train_date[1], datetime):
                str_date = train_date[1].strftime(self.date_format)

            if self.method == "divergence":
                scores_at_t, variables_at_t = divergence_filter.calculate_feature_importance(df_at_t, features)
                features_variables_importances.update({str_date: variables_at_t})
            elif self.method == "permutation":
                scores_at_t = permutation_urf.calculate_feature_importance(
                    self.h2o_context.asH2OFrame(df_at_t), features
                )
            elif self.method == "niv":
                scores_at_t, variables_at_t = niv.calculate_feature_importance(df_at_t, features)
                features_variables_importances.update({str_date: variables_at_t})
            elif self.method == "uc":
                scores_at_t, variables_at_t = uplift_curve.calculate_feature_importance(df_at_t, features)
                features_variables_importances.update({str_date: variables_at_t})
            else:
                print(
                    f"The method {self.method} does not exist. Please choose one of the following methods:"
                    f"{'permutation', 'divergence', 'niv', 'uc'}"
                )
                return feature_importance_scores, features_variables_importances
            scores_at_t.columns = ["feature", str_date]

            if feature_importance_scores.empty:
                feature_importance_scores = scores_at_t.copy()
            else:
                feature_importance_scores = feature_importance_scores.merge(scores_at_t, on="feature")

        # Min-Max Normalization
        feature_importance_scores.iloc[:, 1:] = min_max_normalization(feature_importance_scores)
        # Linear Weighting
        feature_importance_scores = linear_weighting(feature_importance_scores, time_steps)
        feature_importance_scores.reset_index(drop=True, inplace=True)

        return feature_importance_scores, features_variables_importances
