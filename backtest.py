import numpy as np
import pandas as pd
from numba import njit

@njit
def _backtest_loop(pred, p_buy, p_sell, price, bsi, min_prob, leverage, min_hold_bars, bsi_confirm_bars, initial_capital):
    n = len(pred)
    pnl_out = np.zeros(n)
    entry_bar_out = np.full(n, -1, dtype=np.int64)
    exit_bar_out = np.full(n, -1, dtype=np.int64)
    signal_out = np.zeros(n, dtype=np.int64)

    position = 0
    entry_price = 0.0
    entry_bar = -1
    bsi_counter = 0

    for i in range(n):
        if position == 0:
            if pred[i] == 1 and p_buy[i] >= min_prob:
                position, entry_price, entry_bar, bsi_counter = 1, price[i], i, 0
            elif pred[i] == -1 and p_sell[i] >= min_prob:
                position, entry_price, entry_bar, bsi_counter = -1, price[i], i, 0

        elif i - entry_bar >= min_hold_bars:
            bsi_counter = bsi_counter + 1 if (position == 1 and bsi[i] <= 0) or (position == -1 and bsi[i] >= 0) else 0
            if bsi_counter >= bsi_confirm_bars:
                pnl_out[i] = initial_capital * leverage * (price[i] - entry_price) / entry_price * position
                entry_bar_out[i], exit_bar_out[i], signal_out[i] = entry_bar, i, position
                position, bsi_counter = 0, 0

    if position != 0:
        i = n - 1
        pnl_out[i] = initial_capital * leverage * (price[i] - entry_price) / entry_price * position
        entry_bar_out[i], exit_bar_out[i], signal_out[i] = entry_bar, i, position

    return pnl_out, entry_bar_out, exit_bar_out, signal_out

@njit
def _realistic_backtest_loop(pred, p_buy, p_sell, price, bsi, stamps_ns,
                              min_prob, leverage, initial_capital,
                              latency_ns, jitter_ns, jitter_prob,
                              half_spread, slippage_std, partial_fill_prob, partial_fill_min,
                              rejection_prob, borrow_rate_per_sec, commission_per_trade):
    n = len(pred)
    pnl_out = np.zeros(n)
    signal_out = np.zeros(n, dtype=np.int64)
    entry_bar_out = np.full(n, -1, dtype=np.int64)
    fill_fraction_out = np.ones(n)

    position = 0
    entry_price = 0.0
    entry_time_ns = np.int64(0)
    entry_bar = -1
    fill_fraction = 1.0
    entry_signal = 0

    has_pending = False
    pend_fire_ns = np.int64(0)
    pend_action = 0
    pend_entry_price = 0.0
    pend_entry_time = np.int64(0)
    pend_entry_signal = 0

    for i in range(n):
        current_ns = stamps_ns[i]

        if has_pending and current_ns >= pend_fire_ns:
            if i > 0:
                span = float(current_ns - stamps_ns[i - 1])
                frac = max(0.0, min(1.0, float(pend_fire_ns - stamps_ns[i - 1]) / span if span > 0 else 1.0))
                fill_price = price[i - 1] + frac * (price[i] - price[i - 1])
            else:
                fill_price = price[i]
            has_pending = False

            if pend_action != 0:
                if np.random.random() < rejection_prob:
                    continue
                ff = 1.0
                if np.random.random() < partial_fill_prob:
                    ff = partial_fill_min + np.random.random() * (1.0 - partial_fill_min)
                slip = abs(np.random.normal(0.0, slippage_std))
                fill_price += (half_spread + slip) * pend_action
                position = pend_action
                entry_price = fill_price
                fill_fraction = ff
                entry_time_ns = pend_fire_ns
                entry_bar = i
                entry_signal = pend_action
            else:
                slip = abs(np.random.normal(0.0, slippage_std))
                fill_price -= (half_spread + slip) * pend_entry_signal
                borrow_cost = 0.0
                if pend_entry_signal == -1:
                    borrow_cost = initial_capital * leverage * fill_fraction * borrow_rate_per_sec * float(pend_fire_ns - pend_entry_time) / 1e9
                raw_pnl = initial_capital * leverage * fill_fraction * (fill_price - pend_entry_price) / pend_entry_price * pend_entry_signal
                pnl_out[i] = raw_pnl - borrow_cost - commission_per_trade
                signal_out[i] = pend_entry_signal
                entry_bar_out[i] = entry_bar
                fill_fraction_out[i] = fill_fraction
                position, fill_fraction, entry_signal = 0, 1.0, 0

        if not has_pending:
            eff_ns = latency_ns + (int(np.random.random() * jitter_ns) if np.random.random() < jitter_prob else 0)
            if position == 0:
                if pred[i] == 1 and p_buy[i] >= min_prob:
                    has_pending, pend_fire_ns, pend_action = True, current_ns + eff_ns, 1
                    pend_entry_price, pend_entry_time, pend_entry_signal = 0.0, np.int64(0), 0
                elif pred[i] == -1 and p_sell[i] >= min_prob:
                    has_pending, pend_fire_ns, pend_action = True, current_ns + eff_ns, -1
                    pend_entry_price, pend_entry_time, pend_entry_signal = 0.0, np.int64(0), 0
            elif (position == 1 and bsi[i] <= 0) or (position == -1 and bsi[i] >= 0):
                has_pending, pend_fire_ns, pend_action = True, current_ns + eff_ns, 0
                pend_entry_price, pend_entry_time, pend_entry_signal = entry_price, entry_time_ns, entry_signal

    if position != 0:
        i = n - 1
        borrow_cost = 0.0
        if position == -1:
            borrow_cost = initial_capital * leverage * fill_fraction * borrow_rate_per_sec * float(stamps_ns[i] - entry_time_ns) / 1e9
        raw_pnl = initial_capital * leverage * fill_fraction * (price[i] - entry_price) / entry_price * position
        pnl_out[i] = raw_pnl - borrow_cost - commission_per_trade
        signal_out[i] = position
        entry_bar_out[i] = entry_bar
        fill_fraction_out[i] = fill_fraction

    return pnl_out, signal_out, entry_bar_out, fill_fraction_out

def _build_trades(df, exit_bars, pnl_arr, signal_arr, entry_bar_arr, fill_frac_arr=None):
    stamps = df["stamp"].values
    dur_list = []
    for idx in exit_bars:
        eb = entry_bar_arr[idx]
        entry_t = pd.to_datetime(stamps[eb if eb >= 0 else idx])
        dur_list.append((pd.to_datetime(stamps[idx]) - entry_t).total_seconds() / 60)
    pnl_list = pnl_arr[exit_bars]
    result = {
        "stamp": stamps[exit_bars],
        "signal": signal_arr[exit_bars],
        "pnl": pnl_list,
        "duration_min": dur_list,
        "correct": (pnl_list > 0).astype(int),
    }
    if fill_frac_arr is not None:
        result["fill_fraction"] = fill_frac_arr[exit_bars]
    return pd.DataFrame(result)

def backtest(metrics, min_prob=0.6, leverage=1.0, min_hold_bars=0, bsi_confirm_bars=0, initial_capital=1000.0):
    df = metrics.copy().reset_index(drop=True)
    pnl_arr, entry_bar_arr, exit_bar_arr, signal_arr = _backtest_loop(
        df["pred"].to_numpy(dtype=np.float64),
        df["p_buy"].to_numpy(dtype=np.float64),
        df["p_sell"].to_numpy(dtype=np.float64),
        df["price"].to_numpy(dtype=np.float64),
        df["bsi"].to_numpy(dtype=np.float64),
        float(min_prob), float(leverage), int(min_hold_bars), int(bsi_confirm_bars), float(initial_capital)
    )
    exit_bars = np.where(exit_bar_arr >= 0)[0]
    if len(exit_bars) == 0:
        return pd.DataFrame(columns=["stamp", "signal", "pnl", "duration_min", "correct"])
    return _build_trades(df, exit_bars, pnl_arr, signal_arr, entry_bar_arr)

def realistic_backtest(metrics, min_prob=0.6, leverage=1.0,
                       latency_sec=0.05, jitter_sec=0.01, jitter_prob=0.15,
                       half_spread=0.0011, slippage_std=0.0143,
                       partial_fill_prob=0.0, partial_fill_min=0.5,
                       borrow_rate_annual=0.03, rejection_prob=0.001,
                       commission_per_trade=0.00, initial_capital=1000.0):
    df = metrics.copy().reset_index(drop=True)
    pnl_arr, signal_arr, entry_bar_arr, fill_frac_arr = _realistic_backtest_loop(
        df["pred"].to_numpy(dtype=np.float64),
        df["p_buy"].to_numpy(dtype=np.float64),
        df["p_sell"].to_numpy(dtype=np.float64),
        df["price"].to_numpy(dtype=np.float64),
        df["bsi"].to_numpy(dtype=np.float64),
        pd.to_datetime(df["stamp"]).astype(np.int64).to_numpy(),
        float(min_prob), float(leverage), float(initial_capital),
        np.int64(int(latency_sec * 1e9)), np.int64(int(jitter_sec * 1e9)), float(jitter_prob),
        float(half_spread), float(slippage_std), float(partial_fill_prob), float(partial_fill_min),
        float(rejection_prob), float(0), float(commission_per_trade)
    )
    exit_bars = np.where(signal_arr != 0)[0]
    if len(exit_bars) == 0:
        return pd.DataFrame(columns=["stamp", "signal", "pnl", "duration_min", "correct", "fill_fraction"])
    return _build_trades(df, exit_bars, pnl_arr, signal_arr, entry_bar_arr, fill_frac_arr)