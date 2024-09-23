import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download mutual fund data (for example, Vanguard 500 Index Fund Admiral Shares and others)
def download_fund_data(ticker, start='2018-01-01', end='2023-01-01'):
    return yf.download(ticker, start=start, end=end)

funds = {
    'VFIAX': download_fund_data('VFIAX'),
    'VTSAX': download_fund_data('VTSAX'),
    'FZROX': download_fund_data('FZROX')
}

# Calculate Annualized Returns and Volatility
def annualized_return(df):
    daily_return = df['Adj Close'].pct_change().mean()
    return (1 + daily_return) ** 252 - 1

def annualized_volatility(df):
    daily_vol = df['Adj Close'].pct_change().std()
    return daily_vol * np.sqrt(252)

# Sharpe Ratio Calculation
def sharpe_ratio(df, risk_free_rate=0.01):
    daily_return = df['Adj Close'].pct_change().mean()
    daily_volatility = df['Adj Close'].pct_change().std()
    sharpe = (daily_return - risk_free_rate / 252) / daily_volatility * np.sqrt(252)
    return sharpe

# Maximum Drawdown Calculation
def max_drawdown(df):
    roll_max = df['Adj Close'].cummax()
    drawdown = df['Adj Close'] / roll_max - 1.0
    return drawdown.min()

# Sortino Ratio Calculation
def sortino_ratio(df, risk_free_rate=0.01):
    daily_return = df['Adj Close'].pct_change().mean()
    negative_volatility = df['Adj Close'].pct_change()[df['Adj Close'].pct_change() < 0].std()
    sortino = (daily_return - risk_free_rate / 252) / negative_volatility * np.sqrt(252)
    return sortino

# Treynor Ratio Calculation
def treynor_ratio(df, beta, risk_free_rate=0.01):
    daily_return = df['Adj Close'].pct_change().mean()
    treynor = (daily_return - risk_free_rate / 252) / beta
    return treynor * 252  # Annualized Treynor Ratio

# Cumulative Returns Calculation
def cumulative_returns(df):
    return (1 + df['Adj Close'].pct_change()).cumprod() - 1

# Plot Cumulative Returns for all Funds
def plot_cumulative_returns(funds):
    plt.figure(figsize=(12, 6))
    for fund_name, data in funds.items():
        plt.plot(cumulative_returns(data), label=fund_name)
    plt.title("Cumulative Returns of Mutual Funds")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Cumulative Returns and Rolling Sharpe Ratio for each Fund
def plot_fund_performance(df, title):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns', color='blue')
    ax1.plot(cumulative_returns(df), color='blue', label='Cumulative Returns')
    ax1.tick_params(axis='y')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Rolling Sharpe Ratio (252 Days)', color='green')
    ax2.plot(df['Adj Close'].pct_change().rolling(window=252).apply(lambda x: sharpe_ratio(df)), color='green', label='Rolling Sharpe Ratio')
    ax2.tick_params(axis='y')
    
    fig.tight_layout()
    plt.show()

# Plot Heatmap for Fund Comparison (Annual Return, Volatility, Sharpe Ratio, Max Drawdown)
def plot_fund_comparison_with_drawdown(funds):
    comparison_data = {
        'Fund': [],
        'Annualized Return': [],
        'Volatility': [],
        'Sharpe Ratio': [],
        'Max Drawdown': []
    }
    for fund_name, data in funds.items():
        comparison_data['Fund'].append(fund_name)
        comparison_data['Annualized Return'].append(annualized_return(data))
        comparison_data['Volatility'].append(annualized_volatility(data))
        comparison_data['Sharpe Ratio'].append(sharpe_ratio(data))
        comparison_data['Max Drawdown'].append(max_drawdown(data))
    
    comparison_df = pd.DataFrame(comparison_data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(comparison_df.set_index('Fund').T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Mutual Fund Comparison (Annual Return, Volatility, Sharpe Ratio, Max Drawdown)')
    plt.show()

# Plot Rolling Volatility for each Fund
def plot_rolling_volatility(df, window=252):
    df['Rolling Volatility'] = df['Adj Close'].pct_change().rolling(window).std() * np.sqrt(252)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Rolling Volatility'], label='Rolling Volatility')
    plt.title('Rolling Volatility (252 Days)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Heatmap for Fund Comparison with Sortino Ratio
def plot_fund_comparison_with_sortino(funds):
    comparison_data = {
        'Fund': [],
        'Annualized Return': [],
        'Volatility': [],
        'Sharpe Ratio': [],
        'Sortino Ratio': []
    }
    for fund_name, data in funds.items():
        comparison_data['Fund'].append(fund_name)
        comparison_data['Annualized Return'].append(annualized_return(data))
        comparison_data['Volatility'].append(annualized_volatility(data))
        comparison_data['Sharpe Ratio'].append(sharpe_ratio(data))
        comparison_data['Sortino Ratio'].append(sortino_ratio(data))
    
    comparison_df = pd.DataFrame(comparison_data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(comparison_df.set_index('Fund').T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Mutual Fund Comparison (Annual Return, Volatility, Sharpe Ratio, Sortino Ratio)')
    plt.show()

# Plot Heatmap for Fund Comparison with Treynor Ratio
def plot_fund_comparison_with_treynor(funds, betas):
    comparison_data = {
        'Fund': [],
        'Annualized Return': [],
        'Volatility': [],
        'Sharpe Ratio': [],
        'Treynor Ratio': []
    }
    for fund_name, data in funds.items():
        comparison_data['Fund'].append(fund_name)
        comparison_data['Annualized Return'].append(annualized_return(data))
        comparison_data['Volatility'].append(annualized_volatility(data))
        comparison_data['Sharpe Ratio'].append(sharpe_ratio(data))
        comparison_data['Treynor Ratio'].append(treynor_ratio(data, betas[fund_name]))
    
    comparison_df = pd.DataFrame(comparison_data)
    plt.figure(figsize=(10, 6))
    sns.heatmap(comparison_df.set_index('Fund').T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Mutual Fund Comparison (Annual Return, Volatility, Sharpe Ratio, Treynor Ratio)')
    plt.show()

# Correlation Matrix Visualization
def plot_correlation_matrix(funds):
    combined_df = pd.DataFrame()
    for fund_name, data in funds.items():
        combined_df[fund_name] = data['Adj Close']

    correlation_matrix = combined_df.pct_change().corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Mutual Funds')
    plt.show()

# Risk-Reward Scatter Plot
def risk_reward_scatter_plot(funds):
    fig, ax = plt.subplots(figsize=(10, 6))
    for fund_name, data in funds.items():
        ann_return = annualized_return(data)
        ann_volatility = annualized_volatility(data)
        ax.scatter(ann_volatility, ann_return, label=fund_name)
        ax.text(ann_volatility, ann_return, fund_name, fontsize=12)

    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Risk vs. Reward of Mutual Funds')
    ax.grid(True)
    ax.legend()
    plt.show()

# Recommend Funds Based on Risk Tolerance
def recommend_funds(funds, risk_tolerance='medium'):
    recommendations = {}
    for name, data in funds.items():
        ann_return = annualized_return(data)
        ann_vol = annualized_volatility(data)
        sharpe = sharpe_ratio(data)
        
        if risk_tolerance == 'low' and ann_vol < 0.15 and sharpe > 1:
            recommendations[name] = (ann_return, ann_vol, sharpe)
        elif risk_tolerance == 'medium' and 0.15 <= ann_vol <= 0.25:
            recommendations[name] = (ann_return, ann_vol, sharpe)
        elif risk_tolerance == 'high' and ann_vol > 0.25:
            recommendations[name] = (ann_return, ann_vol, sharpe)
    
    print(f"Funds recommended for {risk_tolerance} risk tolerance:")
    for name, (ann_return, ann_vol, sharpe) in recommendations.items():
        print(f"{name}: Annual Return: {ann_return:.2f}, Volatility: {ann_vol:.2f}, Sharpe Ratio: {sharpe:.2f}")

# Usage examples:
# Replace fund ticker and beta values as per your data.
betas = {'VFIAX': 1.00, 'VTSAX': 0.98, 'FZROX': 1.02}  # Example beta values

plot_cumulative_returns(funds)
plot_fund_performance(funds['VFIAX'], "VFIAX Performance")
plot_fund_comparison_with_drawdown(funds)
plot_fund_comparison_with_sortino(funds)
plot_fund_comparison_with_treynor(funds, betas)
plot_correlation_matrix(funds)
risk_reward_scatter_plot(funds)
recommend_funds(funds, risk_tolerance='medium')
