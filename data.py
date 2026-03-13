import math
import pandas as pd
import numpy as np

def compute_sharpe_ratio(trades, risk_free_rate=0.0415):
    if trades.empty:
        return None
    pnl = trades["pnl"]
    sharpe_ratio = ((pnl.mean() - risk_free_rate/252) / pnl.std()) * math.sqrt(252)
    return sharpe_ratio

def remove_outliers(data):
    if len(data) == 0:
        return data
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    return [x for x in data if (q1 - 1.5*iqr) <= x <= (q3 + 1.5*iqr)]

def load_taq_data(csv_path, last_stamp=None, max_chunks=0, resample_freq="1s"):
    usecols = ["DATE","TIME_M","PRICE","SIZE","TR_CORR","TR_SCOND"]
    bars_list = []
    new_chunks_processed = 0

    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=1_000_000)):
        if new_chunks_processed >= max_chunks:
            break

        # Parse timestamp
        chunk["TIME_M"] = chunk["TIME_M"].astype(str).str.zfill(15)
        chunk["stamp"]  = pd.to_datetime(chunk["DATE"] + " " + chunk["TIME_M"],
                                        format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
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

        # compute buy/sell volumes using numpy instead of iloc
        price_diff = chunk["price"].to_numpy()
        price_diff = np.diff(price_diff, prepend=price_diff[0])
        sign = np.sign(price_diff)
        sign[0] = 0  # replace first element directly

        sign = pd.Series(sign).replace(0, np.nan).ffill().fillna(0)
        chunk["buyvolume"]  = np.where(sign > 0, chunk["size"], 0.0)
        chunk["sellvolume"] = np.where(sign < 0, chunk["size"], 0.0)

        # Resample to desired frequency
        bars = chunk.resample(resample_freq, on="stamp").agg(
            price=("price","last"),
            buyvolume=("buyvolume","sum"),
            sellvolume=("sellvolume","sum")
        ).dropna()
        bars_list.append(bars)
        new_chunks_processed += 1

    if not bars_list:
        return pd.DataFrame(columns=["stamp","price","buyvolume","sellvolume"])

    return pd.concat(bars_list).reset_index(drop=False)