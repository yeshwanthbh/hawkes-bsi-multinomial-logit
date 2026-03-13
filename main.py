import os
import pandas as pd
from feature_engineering import compute_bsi_for_metrics
from models import LogisticRegressionModel, MultiLayerPerceptionFFNNModel
from data import load_taq_data, compute_sharpe_ratio
from backtest import backtest, realistic_backtest
from joblib import dump, load
from monte_carlo import monte_carlo_simulation

def get_model(train_metrics, model_file, scaler_file, min_prob=0.7):
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        clf    = load(model_file)
        scaler = load(scaler_file)
        model_train = None
    else:
        model_train = MultiLayerPerceptionFFNNModel(train_metrics, horizon=1)
        clf = model_train.train()
        scaler = model_train.scaler
        dump(clf, model_file)
        dump(scaler, scaler_file)
    return clf, scaler, model_train

def run_backtests(metrics, min_prob=0.7, leverage=1):
    bt_results = backtest(metrics, min_prob=min_prob, leverage=leverage)
    rbt_results = realistic_backtest(metrics, min_prob=min_prob, leverage=leverage)
    return bt_results, rbt_results

def main(csv_path, save_dir, min_prob=0.7, leverage=1):
    print("Loading data and computing features")
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = os.path.join(save_dir, "metrics.parquet")
    model_file   = os.path.join(save_dir, "MLP_model.joblib")
    scaler_file  = os.path.join(save_dir, "scaler.joblib")

    if os.path.exists(metrics_file):
        metrics = pd.read_parquet(metrics_file)
        last_stamp = pd.to_datetime(metrics["stamp"]).max()
    else:
        metrics = pd.DataFrame(columns=["stamp","price","buyvolume","sellvolume","bsi"])
        last_stamp = None

    new_metrics = load_taq_data(csv_path, last_stamp=last_stamp, max_chunks=0)
    print("Columns:", new_metrics.columns.tolist())
    print("Loaded rows:", len(new_metrics))
    if not new_metrics.empty:
        new_metrics = compute_bsi_for_metrics(new_metrics)
        metrics = pd.concat([metrics, new_metrics], ignore_index=True).drop_duplicates(subset="stamp").reset_index(drop=True)
        metrics.to_parquet(metrics_file)

    split_index = int(len(metrics) * 0.8)
    train_metrics = metrics.head(split_index).copy()
    test_metrics  = metrics.tail(len(metrics) - split_index).copy()

    print("Training model")
    clf, scaler, _ = get_model(train_metrics, model_file, scaler_file, min_prob=min_prob)
    model_test = MultiLayerPerceptionFFNNModel(test_metrics, horizon=1, scaler=scaler, fit_scaler=False)

    print("Generating predictions for backtesting")
    model_test.predict(clf, min_prob=min_prob)

    print("Running Backtests")
    bt_results, rbt_results = run_backtests(model_test.metrics, min_prob=min_prob, leverage=leverage)
    backtest_sharpe = compute_sharpe_ratio(bt_results)
    realistic_backtest_sharpe = compute_sharpe_ratio(rbt_results)

    print("Running Monte Carlo Simulation")
    pnl_simulations, accuracy_simulations = monte_carlo_simulation(model_test.metrics)

    return bt_results, rbt_results, backtest_sharpe, realistic_backtest_sharpe

if __name__ == "__main__":
    bt_results, rbt_results, backtest_sharpe, realistic_backtest_sharpe = main(csv_path="path/to/your/TAQ_DATA.csv", save_dir="path/to/your/SavedData")

    print("Backtest Results:")
    print("Total PnL:", bt_results["pnl"].sum())
    print("Total Trades:", len(bt_results))
    print("Correct Trades:", bt_results["correct"].sum())
    print("Accuracy:", bt_results["correct"].mean() * 100 if len(bt_results) > 0 else 0, "%")
    print("Average PnL per Trade:", bt_results["pnl"].mean() if len(bt_results) > 0 else 0)
    print("Average Duration (min):", bt_results["duration_min"].mean() if len(bt_results) > 0 else 0)
    print(bt_results["stamp"].head())
    print(bt_results["stamp"].tail())
    print("Backtest Sharpe Ratio:", backtest_sharpe)

    print("\nRealistic Backtest Results:")
    print("Total PnL:", rbt_results["pnl"].sum())
    print("Total Trades:", len(rbt_results))
    print("Correct Trades:", rbt_results["correct"].sum())
    print("Accuracy:", rbt_results["correct"].mean() * 100 if len(rbt_results) > 0 else 0, "%")
    print("Average PnL per Trade:", rbt_results["pnl"].mean() if len(rbt_results) > 0 else 0)
    print("Average Duration (min):", rbt_results["duration_min"].mean() if len(rbt_results) > 0 else 0)
    print(rbt_results["stamp"].head())
    print(rbt_results["stamp"].tail())
    print("Realistic Backtest Sharpe Ratio:", realistic_backtest_sharpe)

    print("\nMonte Carlo Simulation:")
    print("Monte Carlo Simulation Results:")
    print("Average PnL:", sum(pnl_simulations) / len(pnl_simulations) if len(pnl_simulations) > 0 else 0)
    print("Average Accuracy:", sum(accuracy_simulations) / len(accuracy_simulations) if len(accuracy_simulations) > 0 else 0, "%")
