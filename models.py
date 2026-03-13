import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from feature_engineering import compute_features

class LogisticRegressionModel:
    def __init__(self, metrics: pd.DataFrame, horizon: int, scaler: StandardScaler = None, fit_scaler: bool = True):
        metrics = compute_features(metrics.copy().reset_index(drop=True))
        metrics["future_return"] = metrics["price"].shift(-horizon) - metrics["price"]
        eps = 0.0005
        metrics["y"] = 0
        metrics.loc[metrics["future_return"] >  eps, "y"] =  1
        metrics.loc[metrics["future_return"] < -eps, "y"] = -1
        metrics = metrics.dropna()
        self.metrics = metrics

        X = metrics[["bsi", "ret_1bar", "vol", "vwap_dist"]].values
        if len(X) == 0:
            if fit_scaler:
                raise ValueError("No usable rows remain after feature engineering; need more input data before training.")
            self.X_scaled = np.empty((0, 4), dtype=float)
            self.scaler = scaler
            self.y = np.empty((0,), dtype=int)
            return
        if scaler is None:
            scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)
        self.scaler = scaler
        self.y = metrics["y"].values

    def train(self) -> LogisticRegression:
        clf = LogisticRegression(solver="lbfgs", max_iter=1_000_000_000)
        clf.fit(self.X_scaled, self.y)
        return clf

    @staticmethod
    def decide(row, min_prob: float) -> int:
        if row["p_buy"] >= min_prob: return 1
        if row["p_sell"] >= min_prob: return -1
        return 0

    def predict(self, clf: LogisticRegression, min_prob: float):
        if len(self.X_scaled) == 0:
            self.metrics["p_buy"] = pd.Series(dtype=float)
            self.metrics["p_sell"] = pd.Series(dtype=float)
            self.metrics["p_hold"] = pd.Series(dtype=float)
            self.metrics["pred"] = pd.Series(dtype=int)
            return
        probs = clf.predict_proba(self.X_scaled)
        class_to_col = {c: i for i, c in enumerate(clf.classes_)}
        self.metrics["p_buy"]  = probs[:, class_to_col[ 1]] if 1 in class_to_col else np.zeros(len(probs))
        self.metrics["p_sell"] = probs[:, class_to_col[-1]] if -1 in class_to_col else np.zeros(len(probs))
        self.metrics["p_hold"] = probs[:, class_to_col[ 0]] if 0 in class_to_col else np.zeros(len(probs))
        self.metrics["pred"] = self.metrics.apply(lambda r: LogisticRegressionModel.decide(r, min_prob), axis=1)

class MultiLayerPerceptionFFNNModel:
    def __init__(self, metrics: pd.DataFrame, horizon: int, scaler: StandardScaler = None, fit_scaler: bool = True):
        metrics = compute_features(metrics.copy().reset_index(drop=True))
        metrics["future_return"] = metrics["price"].shift(-horizon) - metrics["price"]
        eps = 0.0005
        metrics["y"] = 0
        metrics.loc[metrics["future_return"] >  eps, "y"] =  1
        metrics.loc[metrics["future_return"] < -eps, "y"] = -1
        metrics = metrics.dropna()
        self.metrics = metrics

        X = metrics[["bsi", "ret_1bar", "vol", "vwap_dist"]].values
        if len(X) == 0:
            if fit_scaler:
                raise ValueError("No usable rows remain after feature engineering; need more input data before training.")
            self.X_scaled = np.empty((0, 4), dtype=float)
            self.scaler = scaler
            self.y = np.empty((0,), dtype=int)
            return
        if scaler is None:
            scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)
        self.scaler = scaler
        self.y = metrics["y"].values

    def train(self) -> MLPClassifier:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=1_000_000_000, random_state=42)
        clf.fit(self.X_scaled, self.y)
        return clf

    @staticmethod
    def decide(row, min_prob: float) -> int:
        if row["p_buy"] >= min_prob: return 1
        if row["p_sell"] >= min_prob: return -1
        return 0

    def predict(self, clf: MLPClassifier, min_prob: float):
        if len(self.X_scaled) == 0:
            self.metrics["p_buy"] = pd.Series(dtype=float)
            self.metrics["p_sell"] = pd.Series(dtype=float)
            self.metrics["p_hold"] = pd.Series(dtype=float)
            self.metrics["pred"] = pd.Series(dtype=int)
            return
        probs = clf.predict_proba(self.X_scaled)
        class_to_col = {c: i for i, c in enumerate(clf.classes_)}
        self.metrics["p_buy"]  = probs[:, class_to_col[ 1]] if 1 in class_to_col else np.zeros(len(probs))
        self.metrics["p_sell"] = probs[:, class_to_col[-1]] if -1 in class_to_col else np.zeros(len(probs))
        self.metrics["p_hold"] = probs[:, class_to_col[ 0]] if 0 in class_to_col else np.zeros(len(probs))
        self.metrics["pred"] = self.metrics.apply(lambda r: MultiLayerPerceptionFFNNModel.decide(r, min_prob), axis=1)