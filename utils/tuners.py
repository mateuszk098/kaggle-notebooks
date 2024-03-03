import logging
import time
from abc import abstractmethod
from collections import namedtuple
from functools import partial
from logging import Logger
from pathlib import Path
from typing import Any, Iterable, Protocol, TypeVar, override

import joblib
import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping
from numpy.typing import ArrayLike
from optuna import samplers
from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, Trial
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)

Model = TypeVar("Model", bound=BaseEstimator)
ModelStudy = namedtuple("ModelStudy", ["model", "hps", "study"])


class Tuner(Protocol[Model]):
    model: Model
    X: ArrayLike
    y: ArrayLike
    logger: Logger

    @abstractmethod
    def objective(self, trial: Trial, seed: int = 42) -> Any: ...

    @abstractmethod
    def define_model(self, trial: Trial, seed: int = 42) -> Model: ...

    def run_study(
        self,
        *,
        direction: str | StudyDirection,
        n_trials: int | None = None,
        timeout: float | None = None,
        seed: int = 42,
    ) -> tuple[Model, dict[str, Any], Study]:
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
        best_model = clone(self.model).set_params(**hps)
        return best_model, hps, study

    def tune(
        self,
        *,
        seeds: Iterable[int],
        direction: str | StudyDirection,
        n_trials: int | None = None,
        timeout: float | None = None,
    ) -> dict[int, ModelStudy]:
        models = dict()
        model_name = self._get_model_name()
        models_path = Path("models/")
        models_path.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            self.logger.info(f"Fine Tuning with Seed: {seed!r}")
            best_model, best_hps, study = self.run_study(
                direction=direction,
                n_trials=n_trials,
                timeout=timeout,
                seed=seed,
            )
            models[seed] = ModelStudy(best_model, best_hps, study)
            model_id = f"{model_name}_{seed}_{time.strftime('run_%Y_%m_%d_%H_%M_%S')}.pkl"
            joblib.dump(best_model, models_path / model_id)
            self.logger.info(f"Saving: {model_id!r}\n")

        return models

    def __call__(
        self,
        seeds: Iterable[int],
        direction: str | StudyDirection,
        n_trials: int | None = None,
        timeout: float | None = None,
    ) -> dict[int, ModelStudy]:
        return self.tune(seeds=seeds, direction=direction, n_trials=n_trials, timeout=timeout)

    def _logging_callback(self, study: Study, frozen_trial: FrozenTrial) -> None:
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        headline = f"Trial: {frozen_trial.number:03} - Best Value: {frozen_trial.value:.5f}"
        hps = frozen_trial.user_attrs | frozen_trial.params
        hps_pattern = "\n".join(f"{' ':>5}{k!r}: {v!r}," for k, v in hps.items())
        if not previous_best_value == study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            self.logger.info(headline + "\n{\n" + hps_pattern + "\n}")

    def _get_model_name(self) -> str:
        if isinstance(self.model, Pipeline):
            return self.model[-1].__class__.__name__
        return self.model.__class__.__name__


class LGBMTuner(Tuner[LGBMClassifier]):  # type: ignore - lgbm IS compatible
    def __init__(self, model: LGBMClassifier, X: ArrayLike, y: ArrayLike) -> None:
        self.model = model
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.logger = logging.getLogger(self.__class__.__name__)

    @override
    def objective(self, trial: Trial, seed: int = 42) -> Any:
        X, y = self.X, self.y
        model = self.define_model(trial, seed)

        y_proba = np.zeros((len(y), len(np.unique(y))), dtype=np.float32)  # type: ignore
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        stopping_rounds = 50
        min_delta = 1e-4

        for train_ids, test_ids in skfold.split(X, y):  # type: ignore
            X_train, y_train = X[train_ids], y[train_ids]  # type: ignore
            X_test, y_test = X[test_ids], y[test_ids]  # type: ignore
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
            )
            model.fit(
                X_train,
                y_train,
                eval_metric="multi_logloss",
                eval_set=[(X_valid, y_valid)],
                callbacks=[early_stopping(stopping_rounds, verbose=False, min_delta=min_delta)],
            )
            y_proba[test_ids] = model.predict_proba(X_test)

        return accuracy_score(y, y_proba.argmax(axis=1))

    @override
    def define_model(self, trial: Trial, seed: int = 42) -> LGBMClassifier:
        frozen_hps = {
            "verbose": -1,
            "random_state": seed,
            "n_estimators": 1000,
            "subsample_freq": 1,
        }
        for k, v in frozen_hps.items():
            trial.set_user_attr(k, v)
        hps = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 32, 512, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 0.5, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5, step=0.1),
        }
        return self.model.set_params(**frozen_hps, **hps)  # type: ignore
