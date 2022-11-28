import h2o  # type: ignore
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import approxCountDistinct
from sklearn.metrics import r2_score  # type: ignore

from upliftml.feature_selection.utils import linear_weighting, min_max_normalization, discretizing


def test_linear_weighting(spark: SparkSession, df_pd_feature_importance_scores_over_time: pd.DataFrame) -> None:
    orig_shape = df_pd_feature_importance_scores_over_time.shape

    pdf = linear_weighting(df_pd_feature_importance_scores_over_time, 3)

    # Check if the new dataframe still has the same number of rows as the original dataframe
    assert df_pd_feature_importance_scores_over_time.shape[0] == orig_shape[0]

    # Check if the new dataframe has one more column than the original dataframe
    assert (df_pd_feature_importance_scores_over_time.shape[1] + 1) == orig_shape[1]

    # Check if the weighted feature importance scores are equal to the first feature importance scores in the original dataframe
    assert pdf["weighted_feature_importance"] == df_pd_feature_importance_scores_over_time["2022-05-01"]


def test_min_max_normalization(spark: SparkSession, df_pd_feature_importance_scores: pd.DataFrame) -> None:
    orig_shape = df_pd_feature_importance_scores.shape

    pdf = min_max_normalization(df_pd_feature_importance_scores)

    # Check if the new dataframe still has the same number of rows as the original dataframe
    assert pdf.shape == orig_shape

    # Check if the range of scores for the features is between 0 and 1
    assert pdf["scores"].between(0, 1)


def test_discretizing(spark: SparkSession, df_spark_binary: pyspark.sql.DataFrame) -> None:
    df_spark_binary = df_spark_binary.drop("propensity", "expected_outcome", "actual_cate")
    features = [col for col in df_spark_binary.columns if "feature" in col]
    n_bins = 10

    df, features_new = discretizing(df_spark_binary, features, n_bins)

    # Check if the number of discretized features equals the number of original features
    assert len(features_new) == len(features)

    # Check if the new dataframe has len(features) more columns
    assert len(df.columns) == len(df_spark_binary.columns) + len(features)

    # Check if the discretized features have the correct distributions
    df_test = df.agg(*(approxCountDistinct(F.col(c)).alias(c) for c in features_new)).toPandas()
    assert df_test.to_numpy()[0].tolist() == [n_bins] * len(features_new)
