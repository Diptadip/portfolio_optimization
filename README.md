
# Portfolio Optimization Using Python

This project accomplishes portfolio optimization by using Monte Carlo Simulations and Sharpe Ratio measures that allow one to judge risk-adjusted returns. Optimizing portfolios for risk-conscious investors.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Portfolio Metrics](#portfolio-metrics)
- [Monte Carlo Simulation](#monte-carlo-simulation)
- [Visualization](#visualization)

## Introduction

Portfolio optimization is a very integral element of financial planning, especially in risk management and maximum return. This project focused on creating an optimized stock portfolio by calculating the most essential metrics such as expected returns, volatility, and the Sharpe Ratio, and then running Monte Carlo simulations to find the best portfolio allocations.

## Features
- Collects stock price data and calculates daily log returns.
- Generates random portfolio allocations and evaluates portfolio performance based on Sharpe Ratio.
- Monte Carlo simulation to generate multiple portfolio combinations.
- Optimizes portfolio allocation to minimize risk and maximize returns.
- Visualizes portfolio performance with Matplotlib.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Diptadip/portfolio_optimization.git
   ```

2. Navigate to the project directory:

   ```bash
   cd portfolio-optimization
   ```

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your stock data CSV file and place it in the `data/` directory.

2. Run the Python script to calculate portfolio metrics and perform Monte Carlo simulations:

   ```bash
   python portfolio_optimizer.py
   ```

3. The results will include:
   - Portfolio returns, volatility, and Sharpe ratio.
   - Optimal portfolio weights for both maximum Sharpe ratio and minimum volatility.
   - A scatter plot visualizing portfolio risk vs. return.

## Dataset

### Dataset Format
The dataset used is taken from the historical data of investing.com and is of the format:

![dataset format](https://github.com/user-attachments/assets/91818cec-48da-4e82-be68-0f81856aae54)

### Dataset Head
The dataset after pivoting is: 

![dataset pivot](https://github.com/user-attachments/assets/00bc6d32-282b-47e8-8aa8-16687113d6ce)

## Portfolio Metrics

### Expected Returns
Calculated by annualizing daily log returns:
```python
exp_ret = np.sum((log_return.mean() * rebalance_weights) * 252)
```
### Expected Volatility
Calculated by using the covariance matrix of log returns:
```python
exp_vol = np.sqrt(np.dot(rebalance_weights.T, np.dot(log_return.cov() * 252, rebalance_weights)))
```

### Sharpe Ratio
Risk-adjusted return calculated as:

```python
sharpe_ratio = exp_ret / exp_vol
```
![sharpe ratio](https://github.com/user-attachments/assets/ff73762b-c3aa-4b65-a0ba-c6c3005588a0)

## Monte Carlo Simulation

The Monte Carlo simulation runs through various random portfolio allocations to find the best one:

```python
for ind in range(num_of_portfolios):
    weights = np.random.random(number_of_symbols)
    weights /= np.sum(weights)
    ret_arr[ind] = np.sum((log_return.mean() * weights) * 252)
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 252, weights)))
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
```

![Monte carlo sim](https://github.com/user-attachments/assets/dcae1619-c114-4991-bc60-96a7d9c8dd49)


### Metrics
The important metrics collected from Monte Carlo simulations are:

![Metrics](https://github.com/user-attachments/assets/ba720107-cf79-45af-b3d8-efb8c6ab5ced)

## Visualization

A scatter plot shows the relationship between risk and return for different portfolio allocations. The red star marks the portfolio with the maximum Sharpe Ratio, and the blue star marks the portfolio with the minimum volatility.

```python
plt.scatter(
    y=simulations_df['Returns'],
    x=simulations_df['Volatility'],
    c=simulations_df['Sharpe Ratio'],
    cmap='RdYlBu'
)
plt.title('Portfolio Returns Vs. Risk')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.scatter(max_sharpe_ratio[1], max_sharpe_ratio[0], marker='*', color='r', s=500)
plt.scatter(min_volatility[1], min_volatility[0], marker='*', color='b', s=500)
plt.show()
```
![plot](https://github.com/user-attachments/assets/05088071-24fc-42c3-bb98-e5dddb837205)

