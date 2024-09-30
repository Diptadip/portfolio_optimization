import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Load the stock price data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing date, symbol, and price columns.
    """
    price_data_frame = pd.read_csv(file_path)
    price_data_frame['date'] = pd.to_datetime(price_data_frame['date'], format='%d-%m-%Y')
    price_data_frame = price_data_frame[['date', 'symbol', 'price']]
    
    # Pivot the data
    price_data_frame = price_data_frame.pivot_table(index='date', columns='symbol', values='price')
    
    return price_data_frame


def calculate_metrics(price_data_frame, symbols):
    """
    Calculate log returns, expected returns, expected volatility, and the Sharpe Ratio.

    Args:
        price_data_frame (pd.DataFrame): DataFrame containing stock prices.
        symbols (list): List of stock symbols in the portfolio.

    Returns:
        dict: Dictionary containing expected returns, volatility, Sharpe ratio, and weights DataFrame.
    """
    log_return = np.log(1 + price_data_frame.pct_change())
    
    # Random portfolio weights
    random_weights = np.random.random(len(symbols))
    rebalance_weights = random_weights / np.sum(random_weights)

    # Expected annualized return and volatility
    exp_ret = np.sum((log_return.mean() * rebalance_weights) * 252)
    exp_vol = np.sqrt(np.dot(rebalance_weights.T, np.dot(log_return.cov() * 252, rebalance_weights)))
    sharpe_ratio = exp_ret / exp_vol
    
    weights_df = pd.DataFrame(data={
        'random_weights': random_weights,
        'rebalance_weights': rebalance_weights
    })

    metrics = {
        'expected_returns': exp_ret,
        'expected_volatility': exp_vol,
        'sharpe_ratio': sharpe_ratio,
        'weights_df': weights_df
    }
    
    return metrics


def monte_carlo_simulation(price_data_frame, symbols, iterations=5000):
    """
    Run a Monte Carlo simulation to calculate optimal portfolio weights.

    Args:
        price_data_frame (pd.DataFrame): DataFrame containing stock prices.
        symbols (list): List of stock symbols in the portfolio.
        iterations (int): Number of Monte Carlo simulations to run.

    Returns:
        pd.DataFrame: DataFrame containing results of the Monte Carlo simulation.
    """
    num_symbols = len(symbols)
    
    log_return = np.log(1 + price_data_frame.pct_change())
    
    all_weights = np.zeros((iterations, num_symbols))
    ret_arr = np.zeros(iterations)
    vol_arr = np.zeros(iterations)
    sharpe_arr = np.zeros(iterations)
    
    for i in range(iterations):
        weights = np.random.random(num_symbols)
        weights /= np.sum(weights)
        
        all_weights[i, :] = weights
        
        ret_arr[i] = np.sum((log_return.mean() * weights) * 252)
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 252, weights)))
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
    
    simulations_df = pd.DataFrame({
        'Returns': ret_arr,
        'Volatility': vol_arr,
        'Sharpe Ratio': sharpe_arr,
        'Portfolio Weights': list(all_weights)
    })

    return simulations_df


def plot_simulation_results(simulations_df, max_sharpe, min_volatility):
    """
    Plot the Monte Carlo simulation results.

    Args:
        simulations_df (pd.DataFrame): DataFrame containing simulation results.
        max_sharpe (pd.Series): Portfolio with the maximum Sharpe Ratio.
        min_volatility (pd.Series): Portfolio with the minimum volatility.
    """
    plt.scatter(simulations_df['Volatility'], simulations_df['Returns'], c=simulations_df['Sharpe Ratio'], cmap='RdYlBu')
    
    plt.title('Portfolio Returns vs Risk')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Returns')
    
    # Highlight the portfolio with the highest Sharpe Ratio
    plt.scatter(max_sharpe['Volatility'], max_sharpe['Returns'], color='r', marker='*', s=500, label='Max Sharpe Ratio')
    
    # Highlight the portfolio with the lowest volatility
    plt.scatter(min_volatility['Volatility'], min_volatility['Returns'], color='b', marker='*', s=500, label='Min Volatility')
    
    plt.legend()
    plt.show()


def main():
    # Stock symbols
    symbols = ['AAPL', 'AMZN', 'MSFT']
    
    # Load stock price data
    price_data_frame = load_data('./data/data.csv')
    
    # Calculate portfolio metrics
    metrics = calculate_metrics(price_data_frame, symbols)
    print('Portfolio Metrics:')
    print(metrics['weights_df'])
    print(f"Expected Returns: {metrics['expected_returns']}")
    print(f"Expected Volatility: {metrics['expected_volatility']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
    
    # Run Monte Carlo simulation
    simulations_df = monte_carlo_simulation(price_data_frame, symbols, iterations=5000)
    
    # Find portfolio with max Sharpe Ratio and min Volatility
    max_sharpe_ratio = simulations_df.loc[simulations_df['Sharpe Ratio'].idxmax()]
    min_volatility = simulations_df.loc[simulations_df['Volatility'].idxmin()]
    
    print("Max Sharpe Ratio Portfolio:")
    print(max_sharpe_ratio)
    
    print("Min Volatility Portfolio:")
    print(min_volatility)
    
    # Plot the results
    plot_simulation_results(simulations_df, max_sharpe_ratio, min_volatility)


if __name__ == "__main__":
    main()
