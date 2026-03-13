from backtest import realistic_backtest

def monte_carlo_simulation(metrics, num_iterations=1000):
    pnl_results = []
    accuracy_results = []
    for i in range(int(num_iterations)):
        trades = realistic_backtest(metrics)
        pnl_results.append(trades["pnl"].sum())
        accuracy_results.append(trades["correct"].mean() * 100)
    return pnl_results, accuracy_results