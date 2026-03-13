import pandas as pd
import numpy as np
from numba import njit

def column_for(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(f"None of {candidates} found in DataFrame")

class HawkesBSI:
    def __init__(self, kappa: float):
        self._kappa = float(kappa)
        self._metrics = None

    def eval(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            times = df.index
        else:
            times = column_for(df, ['stamp', 'time', 'Date', 'date', 'datetime'])

        prices = column_for(df, ['close', 'Close', 'price'])
        buyvol = column_for(df, ['buyvolume', 'BuyVolume'])
        sellvol = column_for(df, ['sellvolume', 'SellVolume'])

        dv = (buyvol - sellvol).to_numpy(dtype=float)
        alpha = np.exp(-self._kappa)
        bsi = self._compute_bsi(dv, alpha)

        self._metrics = pd.DataFrame({
            'stamp': times,
            'price': prices,
            'bsi': bsi,
            'buyvolume': buyvol.values,
            'sellvolume': sellvol.values
        })

        return self._metrics

    @staticmethod
    @njit
    def _compute_bsi(dv: np.ndarray, decay: float) -> np.ndarray:
        out = np.zeros(len(dv), dtype=np.float64)
        bsi = 0.0
        for i in range(len(dv)):
            bsi = bsi * decay + dv[i]
            out[i] = bsi
        return out

def compute_bsi_for_metrics(metrics, kappa=0.1):
    metrics = HawkesBSI(kappa=kappa).eval(metrics)
    for col in ["buyvolume", "sellvolume"]:
        if col not in metrics.columns:
            metrics[col] = 0.0
    return metrics

def compute_features(metrics: pd.DataFrame):
    VOL_WINDOW  = 10
    VWAP_WINDOW = 20

    metrics = metrics.copy()
    metrics["ret_1bar"]  = np.log(metrics["price"]).diff()
    metrics["vol"]       = metrics["ret_1bar"].rolling(VOL_WINDOW).std()
    metrics["vwap"]      = metrics["price"].rolling(VWAP_WINDOW).mean()
    metrics["vwap_dist"] = metrics["price"] - metrics["vwap"]
    return metrics