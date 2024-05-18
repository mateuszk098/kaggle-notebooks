import logging
import sys
import time
from abc import abstractmethod
from collections import namedtuple
from functools import partial
from logging import Logger
from pathlib import Path
from typing import Any, Iterable, Protocol, TypeVar

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from optuna import samplers
from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT, stream=sys.stdout)

Model = TypeVar("Model")
ModelStudy = namedtuple("ModelStudy", ["hps", "study"])


# Example custom metrics for LightGBM and XGBoost.
def lgb_roc_auc_ovr(preds, dataset):
    ground_truth = dataset.get_label()
    roc_auc = roc_auc_score(ground_truth, preds, average="macro", multi_class="ovr")
    return "roc_auc_ovr", roc_auc, True  # Higher is better - True.


def xgb_roc_auc_ovr(preds, dataset):
    ground_truth = dataset.get_label()
    roc_auc = roc_auc_score(ground_truth, preds, average="macro", multi_class="ovr")
    return "roc_auc_ovr", roc_auc


# Optuna Tuner Interface.
class Tuner(Protocol):
    logger: Logger
    show_frozen: bool

    @abstractmethod
    def objective(self, trial: Trial, seed: int = 42) -> Any:
        """Objective to be optimized by Optuna."""

    @abstractmethod
    def define_hps(self, trial: Trial, seed: int = 42) -> dict[str, Any]:
        """Define hyperparameters for the model."""

    def run_study(
        self,
        *,
        direction: str | StudyDirection,
        n_trials: int | None = None,
        timeout: float | None = None,
        seed: int = 42,
    ) -> tuple[dict[str, Any], Study]:
        sampler = samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(
            func=partial(self.objective, seed=seed),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._logging_callback],
            n_jobs=-1,
            gc_after_trial=True,
        )
        hps = study.best_trial.user_attrs | study.best_params
        return hps, study

    def tune(
        self,
        *,
        model_name: str,
        seeds: Iterable[int],
        direction: str | StudyDirection,
        n_trials: int | None = None,
        timeout: float | None = None,
    ) -> dict[int, ModelStudy]:
        models = dict()
        models_path = Path("models/")
        models_path.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            self.logger.info(f"Fine Tuning with Seed: {seed!r}")
            best_hps, study = self.run_study(
                direction=direction, n_trials=n_trials, timeout=timeout, seed=seed
            )
            models[seed] = ModelStudy(best_hps, study)
            model_id = f"{model_name}_{seed}_{time.strftime('run_%Y_%m_%d_%H_%M_%S')}.yaml"
            with open(models_path / model_id, "w") as f:
                yaml.dump(best_hps, f)
            self.logger.info(f"Saving: {model_id!r}\n")

        return models

    def __call__(
        self,
        *,
        model_name: str,
        seeds: Iterable[int],
        direction: str | StudyDirection,
        n_trials: int | None = None,
        timeout: float | None = None,
    ) -> dict[int, ModelStudy]:
        return self.tune(
            model_name=model_name,
            seeds=seeds,
            direction=direction,
            n_trials=n_trials,
            timeout=timeout,
        )

    def _logging_callback(self, study: Study, frozen_trial: FrozenTrial) -> None:
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        headline = f"Trial: {frozen_trial.number:03} - Best Value: {frozen_trial.value:.5f}"
        hps = frozen_trial.params
        if self.show_frozen:
            hps = frozen_trial.user_attrs | hps
        hps_pattern = "\n".join(f"{' ':>5}{k!r}: {v!r}," for k, v in hps.items())
        if not previous_best_value == study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            self.logger.info(headline + "\n{\n" + hps_pattern + "\n}")


NUM_ITERATIONS = 1000
EARLY_STOPPING_ROUNDS = 20
MIN_DELTA = 1e-4


class LGBMTuner(Tuner):
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        preprocess: Pipeline | None = None,
        show_frozen: bool = False,
    ) -> None:
        self.X = X
        self.y = y
        self.preprocess = preprocess
        self.show_frozen = show_frozen
        self.logger = logging.getLogger(self.__class__.__name__)

    def objective(self, trial: Trial, seed: int = 42) -> Any:
        X, y = self.X, self.y
        hps = self.define_hps(trial, seed)
        y_proba = np.zeros((len(y), len(np.unique(y))), dtype=np.float32)
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for train_ids, test_ids in skfold.split(X, y):
            X_train, y_train = X.iloc[train_ids], y[train_ids]
            X_test, y_test = X.iloc[test_ids], y[test_ids]
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
            )

            if self.preprocess is not None:
                X_train = self.preprocess.fit_transform(X_train)
                X_valid = self.preprocess.transform(X_valid)
                X_test = self.preprocess.transform(X_test)

            train_set = lgb.Dataset(X_train, label=y_train)
            valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

            model = lgb.train(
                params=hps,
                train_set=train_set,
                valid_sets=[train_set, valid_set],
                valid_names=["train", "valid"],
                num_boost_round=NUM_ITERATIONS,
                feval=lgb_roc_auc_ovr,  # type: ignore
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=EARLY_STOPPING_ROUNDS,
                        min_delta=MIN_DELTA,
                        first_metric_only=True,  # No matter since we have only one metric.
                        verbose=True,
                    ),
                ],
            )
            y_proba[test_ids] = model.predict(X_test)

        return roc_auc_score(y, y_proba, average="macro", multi_class="ovr")

    def define_hps(self, trial: Trial, seed: int = 42) -> dict[str, Any]:
        frozen_hps = {
            "objective": "multiclass",
            "importance_type": "gain",  # The total gain across all splits the feature is used in.
            "metric": "None",  # We will use custom metric for early stopping.
            "bagging_freq": 1,
            "num_class": 8,
            "verbose": -1,
            "seed": seed,
        }
        for k, v in frozen_hps.items():
            trial.set_user_attr(k, v)
        hps = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 32, 512, step=32),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.5),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.05, 20, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.05, 20, log=True),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 0.8, step=0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.8, step=0.1),
        }
        return frozen_hps | hps
