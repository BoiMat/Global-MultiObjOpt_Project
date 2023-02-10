from utils_rules import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler

def BTC_1d_Dataset(zscore = False):
    
    df = pd.read_csv('../Datasets/Binance_BTCUSDT_d.csv')
    df.drop(['Symbol', 'Unix', 'Volume USDT', 'tradecount'], axis=1, inplace=True)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.sort_values('Date')
    df = AddIndicators(df)
    
    if zscore:
       zscore_col = (df['Close'] - df['Close'].rolling(200).mean()) / df['Close'].rolling(200).std()
    
    df.drop(['Open', 'High', 'Low', 'Date'], axis=1, inplace=True)
    
    df_normalized = df.copy()
    for column in df_normalized.columns:
        df_normalized[column] = (df_normalized[column] -
                            df_normalized[column].mean()) / df_normalized[column].std()

    if zscore:        
       df['zscore'] = zscore_col
       df_normalized['zscore'] = zscore_col
    
    df['Entry_Price'] = df['Close'].shift(-1)
    df_normalized['Entry_Price'] = df['Entry_Price']
 
    #df_normalized['Close'] = df['Close']
    df = df.replace(0, 0.000001)
    df_normalized = df_normalized.replace(0, 0.000001)
    df.dropna(inplace=True)
    df_normalized.dropna(inplace=True)
    df_normalized.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if zscore:
       df.drop(['Close'], axis=1, inplace=True)
       df_normalized.drop(['Close'], axis=1, inplace=True)
    
    return df, df_normalized

def SP500_1d_Dataset(zscore = False):
    df = pd.read_csv('../Datasets/S&P500.csv')
    df.columns = ['Date','Close', 'Open', 'High', 'Low', 'Volume', 'Change']
    df['Change'] = df['Change'].str.replace('%', '')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date')
    df['Open'] = df['Open'].str.replace(',', '').astype(float)
    df['High'] = df['High'].str.replace(',', '').astype(float)
    df['Low'] = df['Low'].str.replace(',', '').astype(float)
    df['Close'] = df['Close'].str.replace(',', '').astype(float)
    df['Change'] = df['Change'].str.replace(',', '').astype(float)
    df = AddIndicators(df)
    df.drop(['Date', 'Volume', 'Open', 'High', 'Low'], axis=1, inplace=True)
    
    if zscore:
       zscore_col = (df['Close'] - df['Close'].rolling(200).mean()) / df['Close'].rolling(200).std()

    df_normalized = df.copy()
    for column in df_normalized.columns:
        df_normalized[column] = (df_normalized[column] -
                            df_normalized[column].mean()) / df_normalized[column].std()
    
    if zscore:        
       df['zscore'] = zscore_col
       df_normalized['zscore'] = zscore_col
    
    df['Entry_Price'] = df['Close'].shift(-1)
    df_normalized['Entry_Price'] = df['Entry_Price']

    #df_normalized['Close'] = df['Close']
    df = df.replace(0, 0.000001)
    df_normalized = df_normalized.replace(0, 0.000001)
    df.dropna(inplace=True)
    df_normalized.dropna(inplace=True)
    df_normalized.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if zscore:
       df.drop(['Close'], axis=1, inplace=True)
       df_normalized.drop(['Close'], axis=1, inplace=True)
        
    return df, df_normalized

def plot_results(prices, buy_ops, sell_ops, title="_ Operations", save=False, name="BTC_"):
   plt.figure(figsize=(18,10))
   plt.plot(prices, label='Price')
   plt.xlabel('Date (days)', fontsize=20)
   plt.ylabel('Close Price USD ($)', fontsize=20)
   plt.title(title, fontsize=22)
   plt.xticks(fontsize=18)
   plt.yticks(fontsize=18)
   plt.scatter(buy_ops, prices[buy_ops], color='green', label='Buy', marker='^', alpha=1)
   plt.scatter(sell_ops, prices[sell_ops], color='red', label='Sell', marker='v', alpha=1)
   plt.legend(fontsize=18)
   if save:
      plt.savefig('images/' + name + '.png')