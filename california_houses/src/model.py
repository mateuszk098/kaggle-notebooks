import os

import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # No *args or **kwargs.
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # Eothing else to do.

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]


def stratified_train_test_split(housing):

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):  # type: ignore
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set  # type: ignore


def pipeline_preprocessing(num_attribs, cat_attribs):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    return full_pipeline


def save_model(forest_reg):

    model_dir = os.path.join("..", "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join("..", "models", "forest_reg.pkl")
    joblib.dump(forest_reg, model_path)


def save_test_sets(test_set_labels, test_set_prepared):
    test_set_to_csv = pd.DataFrame(test_set_prepared)  # type: ignore
    test_set_path = os.path.join("..", "data", "test_set.csv")
    test_set_to_csv.to_csv(test_set_path, index=False)

    test_set_labels_to_csv = pd.DataFrame(test_set_labels)
    test_set_labels_path = os.path.join("..", "data", "test_set_labels.csv")
    test_set_labels_to_csv.to_csv(test_set_labels_path, index=False)


def main():

    data_path = os.path.join("..", "data", "housing.csv")
    housing = pd.read_csv(data_path, engine="c")

    housing_train_set, housing_test_set = stratified_train_test_split(housing)

    train_set = housing_train_set.drop("median_house_value", axis=1)  # Remove labels.
    train_set_labels = housing_train_set["median_house_value"].copy()

    num_attribs = list(train_set.select_dtypes(include=["number"]))
    cat_attribs = list(train_set.select_dtypes(include=["object"]))
    full_pipeline = pipeline_preprocessing(num_attribs, cat_attribs)

    train_set_prepared = full_pipeline.fit_transform(train_set)

    forest_reg = RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)
    forest_reg.fit(train_set_prepared, train_set_labels)
    save_model(forest_reg)

    test_set = housing_test_set.drop("median_house_value", axis=1)
    test_set_labels = housing_test_set["median_house_value"].copy()
    test_set_prepared = full_pipeline.transform(test_set)
    save_test_sets(test_set_labels, test_set_prepared)


if __name__ == "__main__":
    main()
