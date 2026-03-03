import os
import random
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from joblib import dump, load

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
            'bsi': bsi
        })

        return self._metrics

    @staticmethod
    def _compute_bsi(dv: np.ndarray, decay: float) -> np.ndarray:
        out = np.zeros_like(dv, dtype=float)
        bsi = 0.0
        for i in range(len(dv)):
            bsi = bsi * decay + dv[i]
            out[i] = bsi
        return out

class LogisticRegressionModel:
    VOL_WINDOW  = 10
    VWAP_WINDOW = 20

    def __init__(self, metrics: pd.DataFrame, horizon: int, scaler: StandardScaler = None, fit_scaler: bool = True):

        metrics = metrics.copy().reset_index(drop=True)

        metrics["future_return"] = metrics["price"].shift(-horizon) - metrics["price"]
        eps = 0.0005
        metrics["y"] = 0
        metrics.loc[metrics["future_return"] >  eps, "y"] =  1
        metrics.loc[metrics["future_return"] < -eps, "y"] = -1

        metrics["ret_1m"]    = np.log(metrics["price"]).diff()
        metrics["vol"]       = metrics["ret_1m"].rolling(self.VOL_WINDOW).std()
        metrics["vwap"]      = metrics["price"].rolling(self.VWAP_WINDOW).mean()
        metrics["vwap_dist"] = metrics["price"] - metrics["vwap"]

        metrics = metrics.dropna()
        self.metrics = metrics

        X = metrics[["bsi", "ret_1m", "vol", "vwap_dist"]].values

        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            self.X_scaled = scaler.fit_transform(X)
        else:
            self.X_scaled = scaler.transform(X)

        self.scaler = scaler
        self.y      = metrics["y"].values

    def train(self) -> LogisticRegression:
        clf = LogisticRegression(solver="lbfgs", max_iter=1_000_000_000)
        clf.fit(self.X_scaled, self.y)
        return clf

    @staticmethod
    def decide(row, min_prob: float) -> int:
        if row["p_buy"]  >= min_prob: return  1
        if row["p_sell"] >= min_prob: return -1
        return 0

    def predict(self, clf: LogisticRegression, min_prob: float):
        probs        = clf.predict_proba(self.X_scaled)
        class_to_col = {c: i for i, c in enumerate(clf.classes_)}

        self.metrics["p_buy"]  = probs[:, class_to_col[ 1]] if  1 in class_to_col else np.zeros(len(probs))
        self.metrics["p_sell"] = probs[:, class_to_col[-1]] if -1 in class_to_col else np.zeros(len(probs))
        self.metrics["p_hold"] = probs[:, class_to_col[ 0]] if  0 in class_to_col else np.zeros(len(probs))

        self.metrics["pred"] = self.metrics.apply(lambda r: LogisticRegressionModel.decide(r, min_prob), axis=1)

        mask    = (self.metrics["p_buy"] >= min_prob) | (self.metrics["p_sell"] >= min_prob)
        total   = mask.sum()
        correct = (self.metrics.loc[mask, "pred"] == self.metrics.loc[mask, "y"]).sum()
        if total > 0:
            print(f"In-sample signal accuracy (threshold={min_prob}): {correct / total:.2%}  ({total} signals)")
        else:
            print("No confident predictions.")

def remove_outliers(data):
    if len(data) == 0:
        return data
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    return [x for x in data if (q1 - 1.5 * iqr) <= x <= (q3 + 1.5 * iqr)]

def fetch_taq_spy_trades(csv_path, bar_freq="1ms", start_date=None, end_date=None, max_chunks=None):
    print("Reading CSV in chunks...")
    usecols    = ["DATE","TIME_M","PRICE","SIZE","TR_CORR","TR_SCOND"]
    chunk_size = 1_000_000
    start_ts   = pd.to_datetime(start_date) if start_date else None
    end_ts     = pd.to_datetime(end_date)   if end_date   else None
    bars_list  = []

    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)):
        if max_chunks is not None and i >= max_chunks:
            print(f"Stopping after {max_chunks} chunks.")
            break
        print(f"Processing chunk {i + 1}")
        chunk["TIME_M"] = chunk["TIME_M"].astype(str).str.zfill(15)
        chunk["stamp"]  = pd.to_datetime(
            chunk["DATE"] + " " + chunk["TIME_M"],
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )
        chunk = chunk.dropna(subset=["stamp"])
        chunk = chunk[chunk["TR_CORR"].astype(int) == 0]
        chunk = chunk[~chunk["TR_SCOND"].isin(["FT","U","Z"])]
        if start_ts is not None: chunk = chunk[chunk["stamp"] >= start_ts]
        if end_ts   is not None: chunk = chunk[chunk["stamp"] <= end_ts]
        if chunk.empty: continue

        chunk      = chunk.sort_values("stamp")
        chunk["price"] = chunk["PRICE"].astype(float)
        chunk["size"]  = chunk["SIZE"].astype(float)

        price_diff = chunk["price"].diff()
        sign       = np.sign(price_diff)
        sign.iloc[0] = 0
        sign       = sign.replace(0, np.nan).ffill().fillna(0)
        chunk["buyvolume"]  = np.where(sign > 0, chunk["size"], 0.0)
        chunk["sellvolume"] = np.where(sign < 0, chunk["size"], 0.0)

        bars = chunk.resample(bar_freq, on="stamp").agg(
            price=("price","last"), buyvolume=("buyvolume","sum"), sellvolume=("sellvolume","sum")
        ).dropna()
        bars_list.append(bars)

    if not bars_list:
        print("No data found for date range.")
        return pd.DataFrame(columns=["stamp","price","buyvolume","sellvolume"])

    print("Combining bars...")
    bars = (pd.concat(bars_list)
              .groupby("stamp")
              .agg(price=("price","last"), buyvolume=("buyvolume","sum"), sellvolume=("sellvolume","sum"))
              .reset_index())
    print(f"Final bars: {len(bars)}")
    return bars

def backtest(metrics, min_prob=0.6, leverage=1.0, min_hold_bars=10, bsi_confirm_bars=3):
    initial_capital  = 1000.0
    rows             = metrics.reset_index(drop=True)
    n                = len(rows)

    position         = 0
    entry_price      = 0.0
    entry_time       = None
    entry_bar        = -1
    total_trades     = 0
    bsi_exit_counter = 0

    pnl_list   = []
    timestamps = []
    durations  = []
    signals    = []
    outcomes   = []

    i = 0
    while i < n:
        row          = rows.iloc[i]
        signal       = row["pred"]
        price        = row["price"]
        bsi          = row["bsi"]
        current_time = pd.to_datetime(row["stamp"])

        if position == 0:
            if signal == 1 and row["p_buy"] >= min_prob:
                position         = 1
                entry_price      = price
                entry_time       = current_time
                entry_bar        = i
                bsi_exit_counter = 0
                total_trades    += 1
            elif signal == -1 and row["p_sell"] >= min_prob:
                position         = -1
                entry_price      = price
                entry_time       = current_time
                entry_bar        = i
                bsi_exit_counter = 0
                total_trades    += 1

        elif position == 1:
            bars_held = i - entry_bar
            if bars_held >= min_hold_bars:
                if bsi <= 0:
                    bsi_exit_counter += 1
                else:
                    bsi_exit_counter  = 0
                if bsi_exit_counter >= bsi_confirm_bars:
                    pnl = initial_capital * leverage * (price - entry_price) / entry_price
                    pnl_list.append(pnl)
                    timestamps.append(row["stamp"])
                    durations.append((current_time - entry_time).total_seconds() / 60)
                    signals.append(1)
                    outcomes.append(1 if pnl > 0 else 0)
                    position         = 0
                    bsi_exit_counter = 0

        elif position == -1:
            bars_held = i - entry_bar
            if bars_held >= min_hold_bars:
                if bsi >= 0:
                    bsi_exit_counter += 1
                else:
                    bsi_exit_counter  = 0
                if bsi_exit_counter >= bsi_confirm_bars:
                    pnl = initial_capital * leverage * (entry_price - price) / entry_price
                    pnl_list.append(pnl)
                    timestamps.append(row["stamp"])
                    durations.append((current_time - entry_time).total_seconds() / 60)
                    signals.append(-1)
                    outcomes.append(1 if pnl > 0 else 0)
                    position         = 0
                    bsi_exit_counter = 0

        i += 1

    if position != 0:
        final_row   = rows.iloc[-1]
        final_price = final_row["price"]
        final_time  = pd.to_datetime(final_row["stamp"])
        pnl = initial_capital * leverage * (
            (final_price - entry_price) if position == 1 else (entry_price - final_price)
        ) / entry_price
        pnl_list.append(pnl)
        timestamps.append(final_row["stamp"])
        durations.append((final_time - entry_time).total_seconds() / 60)
        signals.append(position)
        outcomes.append(1 if pnl > 0 else 0)

    total_pnl        = sum(pnl_list)
    avg_duration_sec = np.mean(remove_outliers(durations)) * 60 if durations else 0.0
    correct_trades   = sum(outcomes)
    n_trades         = len(pnl_list)
    accuracy         = correct_trades / n_trades * 100 if n_trades > 0 else 0.0

    print(f"Total PnL:              ${total_pnl:.2f}")
    print(f"Average position duration: {avg_duration_sec:.2f}s")
    print(f"Total trades:           {n_trades}")
    print(f"Correct trades:         {correct_trades}")
    print(f"Backtest accuracy:      {accuracy:.2f}%")

    trades_df = pd.DataFrame({
        "stamp":        timestamps,
        "signal":       signals,
        "pnl":          pnl_list,
        "duration_min": durations,
        "correct":      outcomes,
    })
    print("\nAll backtest trades:")
    print(trades_df)
    return trades_df

def realistic_backtest(
    metrics,
    min_prob             = 0.6,
    leverage             = 1.0,
    latency_sec          = 0.05,   # real seconds of latency (was latency_bars)
    jitter_sec           = 0.01,   # jitter std in real seconds (was jitter_bars)
    jitter_prob          = 0.15,
    half_spread          = 0.0011, # 0.02 bps — SPY NBBO half-spread
    slippage_std         = 0.0143, # 0.26 bps — SPY market impact
    partial_fill_prob    = 0.0,    # SPY too liquid for partial fills
    partial_fill_min     = 0.5,
    borrow_rate_annual   = 0.03,
    rejection_prob       = 0.001,  # 0.1% — very rare for SPY
    feed_gap_prob        = 0.002,
    feed_gap_bars        = 3,
    commission_per_trade = 0.00,
):
    initial_capital      = 1000.0
    borrow_rate_per_sec  = borrow_rate_annual / (252 * 6.5 * 3600)

    rows = metrics.reset_index(drop=True)
    n    = len(rows)

    position      = 0
    entry_price   = 0.0
    fill_fraction = 1.0
    entry_time    = None
    entry_signal  = 0
    total_trades  = 0

    pnl_list       = []
    timestamps     = []
    durations      = []
    signals        = []
    outcomes       = []
    fill_fractions = []
    rejections     = 0
    gap_bars_left  = 0
    # pending = (fire_time, action, ref_entry_price, ref_entry_time, ref_signal)
    pending        = None

    i = 0
    while i < n:

        if gap_bars_left > 0:
            gap_bars_left -= 1
            i += 1
            continue
        if random.random() < feed_gap_prob:
            gap_bars_left = feed_gap_bars - 1
            i += 1
            continue

        row          = rows.iloc[i]
        signal       = row["pred"]
        price        = row["price"]
        bsi          = row["bsi"]
        current_time = pd.to_datetime(row["stamp"])

        if pending is not None and current_time >= pending[0]:
            fire_time, action, ref_entry_price, ref_entry_time, ref_signal = pending

            if i > 0:
                prev_price = rows.iloc[i - 1]["price"]
                prev_time  = pd.to_datetime(rows.iloc[i - 1]["stamp"])
                span = (current_time - prev_time).total_seconds()
                frac = (fire_time - prev_time).total_seconds() / span if span > 0 else 1.0
                frac = max(0.0, min(1.0, frac))
                fill_price = prev_price + frac * (price - prev_price)
            else:
                fill_price = price

            pending = None

            if action in ("enter_long", "enter_short"):
                if random.random() < rejection_prob:
                    rejections += 1
                    i += 1
                    continue

                frac_fill = 1.0
                if random.random() < partial_fill_prob:
                    frac_fill = random.uniform(partial_fill_min, 1.0)

                slippage = abs(random.gauss(0, slippage_std))
                if action == "enter_long":
                    fill_price  += half_spread + slippage
                    position     = 1
                    entry_signal = 1
                else:
                    fill_price  -= half_spread + slippage
                    position     = -1
                    entry_signal = -1

                entry_price   = fill_price
                fill_fraction = frac_fill
                entry_time    = fire_time
                total_trades += 1

            elif action == "exit":
                slippage = abs(random.gauss(0, slippage_std))
                if ref_signal == 1:
                    fill_price -= half_spread + slippage
                else:
                    fill_price += half_spread + slippage

                borrow_cost = 0.0
                if ref_signal == -1 and ref_entry_time is not None:
                    hold_seconds = (fire_time - ref_entry_time).total_seconds()
                    borrow_cost  = (initial_capital * leverage * fill_fraction
                                    * borrow_rate_per_sec * hold_seconds)

                if ref_signal == 1:
                    raw_pnl = (initial_capital * leverage * fill_fraction
                               * (fill_price - ref_entry_price) / ref_entry_price)
                else:
                    raw_pnl = (initial_capital * leverage * fill_fraction
                               * (ref_entry_price - fill_price) / ref_entry_price)

                pnl = raw_pnl - borrow_cost - commission_per_trade

                pnl_list.append(pnl)
                timestamps.append(fire_time)
                durations.append((fire_time - ref_entry_time).total_seconds() / 60)
                signals.append(ref_signal)
                outcomes.append(1 if pnl > 0 else 0)
                fill_fractions.append(fill_fraction)
                position      = 0
                fill_fraction = 1.0
                entry_signal  = 0

        if pending is None:
            if position == 0:
                if signal == 1 and row["p_buy"] >= min_prob:
                    eff = latency_sec + (abs(random.gauss(0, jitter_sec)) if random.random() < jitter_prob else 0)
                    pending = (current_time + pd.Timedelta(seconds=eff), "enter_long", None, None, None)
                elif signal == -1 and row["p_sell"] >= min_prob:
                    eff = latency_sec + (abs(random.gauss(0, jitter_sec)) if random.random() < jitter_prob else 0)
                    pending = (current_time + pd.Timedelta(seconds=eff), "enter_short", None, None, None)

            elif position == 1 and bsi <= 0:
                eff = latency_sec + (abs(random.gauss(0, jitter_sec)) if random.random() < jitter_prob else 0)
                pending = (current_time + pd.Timedelta(seconds=eff), "exit", entry_price, entry_time, entry_signal)

            elif position == -1 and bsi >= 0:
                eff = latency_sec + (abs(random.gauss(0, jitter_sec)) if random.random() < jitter_prob else 0)
                pending = (current_time + pd.Timedelta(seconds=eff), "exit", entry_price, entry_time, entry_signal)

        i += 1

    final_row   = rows.iloc[-1]
    final_price = final_row["price"]
    final_time  = pd.to_datetime(final_row["stamp"])

    if pending is not None:
        fire_time, action, ref_entry_price, ref_entry_time, ref_signal = pending
        if action == "exit":
            borrow_cost = 0.0
            if ref_signal == -1 and ref_entry_time is not None:
                hold_seconds = (final_time - ref_entry_time).total_seconds()
                borrow_cost  = (initial_capital * leverage * fill_fraction
                                * borrow_rate_per_sec * hold_seconds)
            raw_pnl = initial_capital * leverage * fill_fraction * (
                (final_price - ref_entry_price) if ref_signal == 1
                else (ref_entry_price - final_price)
            ) / ref_entry_price
            pnl = raw_pnl - borrow_cost - commission_per_trade
            pnl_list.append(pnl)
            timestamps.append(final_time)
            durations.append((final_time - ref_entry_time).total_seconds() / 60)
            signals.append(ref_signal)
            outcomes.append(1 if pnl > 0 else 0)
            fill_fractions.append(fill_fraction)
            position = 0

    if position != 0:
        borrow_cost = 0.0
        if position == -1 and entry_time is not None:
            hold_seconds = (final_time - entry_time).total_seconds()
            borrow_cost  = (initial_capital * leverage * fill_fraction
                            * borrow_rate_per_sec * hold_seconds)
        raw_pnl = initial_capital * leverage * fill_fraction * (
            (final_price - entry_price) if position == 1
            else (entry_price - final_price)
        ) / entry_price
        pnl = raw_pnl - borrow_cost - commission_per_trade
        pnl_list.append(pnl)
        timestamps.append(final_time)
        durations.append((final_time - entry_time).total_seconds() / 60)
        signals.append(position)
        outcomes.append(1 if pnl > 0 else 0)
        fill_fractions.append(fill_fraction)

    total_pnl        = sum(pnl_list)
    avg_duration_sec = np.mean(remove_outliers(durations)) * 60 if durations else 0.0
    correct_trades   = sum(outcomes)
    n_trades         = len(pnl_list)
    accuracy         = correct_trades / n_trades * 100 if n_trades > 0 else 0.0
    avg_fill         = np.mean(fill_fractions) * 100 if fill_fractions else 100.0

    print(f"[realistic_backtest | latency={latency_sec*1000:.0f}ms base | jitter={jitter_sec*1000:.0f}ms @ {jitter_prob*100:.0f}% prob]")
    print(f"  spread=+-${half_spread:.4f} | slippage=${slippage_std:.4f} | borrow={borrow_rate_annual*100:.1f}%/yr")
    print(f"  commission=${commission_per_trade:.2f}/trade | partial_fill={partial_fill_prob*100:.0f}% | rejection={rejection_prob*100:.1f}% | feed_gap={feed_gap_prob*100:.2f}%")
    print(f"──────────────────────────────────────────────────")
    print(f"  Total PnL:              ${total_pnl:.2f}")
    print(f"  Total trades:           {n_trades}")
    print(f"  Correct trades:         {correct_trades}")
    print(f"  Accuracy:               {accuracy:.2f}%")
    print(f"  Avg position duration:  {avg_duration_sec:.2f}s")
    print(f"  Avg fill fraction:      {avg_fill:.1f}%")
    print(f"  Orders rejected:        {rejections}")
    print(f"──────────────────────────────────────────────────")

    trades_df = pd.DataFrame({
        "stamp":         timestamps,
        "signal":        signals,
        "pnl":           pnl_list,
        "duration_min":  durations,
        "correct":       outcomes,
        "fill_fraction": fill_fractions,
    })
    print("\nAll realistic backtest trades:")
    print(trades_df)
    return trades_df

def load_new_data(csv_path, last_stamp=None, max_chunks=0, resample_freq="1s"):
    usecols = ["DATE","TIME_M","PRICE","SIZE","TR_CORR","TR_SCOND"]
    bars_list = []
    new_chunks_processed = 0

    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=1_000_000)):
        if new_chunks_processed >= max_chunks:
            break

        print(f"Reading chunk {i+1}...")
        chunk["TIME_M"] = chunk["TIME_M"].astype(str).str.zfill(15)
        chunk["stamp"]  = pd.to_datetime(
            chunk["DATE"] + " " + chunk["TIME_M"],
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )
        chunk = chunk.dropna(subset=["stamp"])
        chunk = chunk[chunk["TR_CORR"].astype(int) == 0]
        chunk = chunk[~chunk["TR_SCOND"].isin(["FT","U","Z"])]
        chunk = chunk.sort_values("stamp")
        chunk["price"] = chunk["PRICE"].astype(float)
        chunk["size"]  = chunk["SIZE"].astype(float)

        if last_stamp is not None:
            chunk = chunk[chunk["stamp"] > last_stamp]
        if chunk.empty:
            continue

        # Compute buy/sell volume
        price_diff = chunk["price"].diff()
        sign       = np.sign(price_diff)
        sign.iloc[0] = 0
        sign = sign.replace(0, np.nan).ffill().fillna(0)
        chunk["buyvolume"]  = np.where(sign > 0, chunk["size"], 0.0)
        chunk["sellvolume"] = np.where(sign < 0, chunk["size"], 0.0)

        # Resample
        bars = chunk.resample(resample_freq, on="stamp").agg(
            price=("price","last"),
            buyvolume=("buyvolume","sum"),
            sellvolume=("sellvolume","sum")
        ).dropna()
        bars_list.append(bars)
        new_chunks_processed += 1
        print(f"Processed new chunk {new_chunks_processed}")

    if not bars_list:
        return pd.DataFrame(columns=["stamp","price","buyvolume","sellvolume"])

    new_metrics = pd.concat(bars_list).reset_index()
    print(f"Loaded {len(new_metrics)} new bars.")
    return new_metrics

def compute_bsi_for_metrics(metrics, kappa=0.1):
    metrics = HawkesBSI(kappa=kappa).eval(metrics)
    for col in ["buyvolume","sellvolume"]:
        if col not in metrics.columns:
            metrics[col] = 0.0
    return metrics

def get_model(train_metrics, model_file, scaler_file, min_prob=0.7):
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        print("Loading existing model and scaler...")
        clf    = load(model_file)
        scaler = load(scaler_file)
        model_train = LogisticRegressionModel(train_metrics, horizon=1, scaler=scaler, fit_scaler=False)
        model_train.predict(clf, min_prob=min_prob)
    else:
        print("Training logistic regression model...")
        model_train = LogisticRegressionModel(train_metrics, horizon=1)
        clf         = model_train.train()
        scaler      = model_train.scaler
        dump(clf, model_file)
        dump(scaler, scaler_file)
        print("Model and scaler saved.")
    return clf, scaler, model_train

def run_backtests(metrics, min_prob=0.7, leverage=1):
    print("Running Backtest...")
    bt_results = backtest(metrics, min_prob=min_prob, leverage=leverage)
    print("Running Realistic Backtest...")
    rbt_results = realistic_backtest(
        metrics,
        min_prob=min_prob,
        leverage=leverage,
        latency_sec=0.01,
        jitter_sec=0.01,
        jitter_prob=0.15,
        half_spread=0.0011,
        slippage_std=0.0143,
        partial_fill_prob=0.0,
        rejection_prob=0.001,
        feed_gap_prob=0.002,
        commission_per_trade=0.0
    )
    return bt_results, rbt_results

def main():
    save_dir     = r"C:\QuantV7\SavedData"
    metrics_file = os.path.join(save_dir, "metrics.parquet")
    model_file   = os.path.join(save_dir, "lr_model.joblib")
    scaler_file  = os.path.join(save_dir, "scaler.joblib")
    csv_path     = r"C:\QuantV7\SPY Data\j1woghljohrbmkdj.csv"

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(metrics_file):
        metrics    = pd.read_parquet(metrics_file)
        last_stamp = pd.to_datetime(metrics["stamp"]).max()
        print(f"Loaded existing metrics ({len(metrics)} rows). Last stamp: {last_stamp}")
    else:
        metrics    = pd.DataFrame(columns=["stamp","price","buyvolume","sellvolume","bsi"])
        last_stamp = None
        print("No existing metrics found. Starting fresh.")

    new_metrics = load_new_data(csv_path, last_stamp=last_stamp, max_chunks=0)
    if not new_metrics.empty:
        print("Computing Hawkes BSI...")
        new_metrics = compute_bsi_for_metrics(new_metrics)
        metrics = pd.concat([metrics, new_metrics], ignore_index=True).drop_duplicates(subset="stamp").reset_index(drop=True)
        metrics.to_parquet(metrics_file)
        new_data_added = True
        print(f"Updated metrics saved ({len(metrics)} rows).")
    else:
        new_data_added = False
        print("No new rows to process.")

    split_index       = int(len(metrics) * 0.5)
    train_metrics     = metrics.iloc[:split_index].copy()
    test_metrics_full = metrics.iloc[split_index:].copy()
    test_metrics      = test_metrics_full.iloc[:1_000_000].copy()  # can adjust size

    clf, scaler, _ = get_model(train_metrics, model_file, scaler_file, min_prob=0.7)

    model_test = LogisticRegressionModel(test_metrics, horizon=1, scaler=scaler, fit_scaler=False)
    model_test.predict(clf, min_prob=0.7)

    bt_results, rbt_results = run_backtests(model_test.metrics, min_prob=0.7, leverage=1)
    print("--- Backtests Completed ---")

if __name__ == "__main__":
    main()