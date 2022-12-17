from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler

def BTC_1d_Dataset():
    
    df = pd.read_csv('Binance_BTCUSDT_d.csv')
    df.drop(['symbol', 'unix', 'Volume USDT', 'tradecount'], axis=1, inplace=True)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.sort_values('Date')
    df = AddIndicators(df)
    
    # normalize the dataset between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_normalized = df.drop(['Open', 'High', 'Low', 'Date'], axis=1)
    df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)

    df['day-1'] = df['Close'].shift(1)
    df['day-2'] = df['Close'].shift(2)
    df['day-3'] = df['Close'].shift(3)
    df['day-4'] = df['Close'].shift(4)
    df['day-5'] = df['Close'].shift(5)
    df['day-6'] = df['Close'].shift(6)
    df['day-7'] = df['Close'].shift(7)
    
    df_normalized['day-1'] = df_normalized['Close'].shift(1)
    df_normalized['day-2'] = df_normalized['Close'].shift(2)
    df_normalized['day-3'] = df_normalized['Close'].shift(3)
    df_normalized['day-4'] = df_normalized['Close'].shift(4)
    df_normalized['day-5'] = df_normalized['Close'].shift(5)
    df_normalized['day-6'] = df_normalized['Close'].shift(6)
    df_normalized['day-7'] = df_normalized['Close'].shift(7)

    #df_normalized['Close'] = df['Close']
    df.dropna(inplace=True)
    df_normalized.dropna(inplace=True)
    df_normalized.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # move 'Close' column to last column
    cols = df_normalized.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df[cols]
    df_normalized = df_normalized[cols]
    
    return df, df_normalized

def SP500_1d_Dataset():
    df = pd.read_csv('S&P500.csv')
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
    df.drop(['Date', 'Volume'], axis=1, inplace=True)

    # normalize the dataset between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_normalized = df.drop(['Open', 'High', 'Low'], axis=1)
    df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)
    
    df['day-1'] = df['Close'].shift(1)
    df['day-2'] = df['Close'].shift(2)
    df['day-3'] = df['Close'].shift(3)
    df['day-4'] = df['Close'].shift(4)
    df['day-5'] = df['Close'].shift(5)
    df['day-6'] = df['Close'].shift(6)
    df['day-7'] = df['Close'].shift(7)
    
    df_normalized['day-1'] = df_normalized['Close'].shift(1)
    df_normalized['day-2'] = df_normalized['Close'].shift(2)
    df_normalized['day-3'] = df_normalized['Close'].shift(3)
    df_normalized['day-4'] = df_normalized['Close'].shift(4)
    df_normalized['day-5'] = df_normalized['Close'].shift(5)
    df_normalized['day-6'] = df_normalized['Close'].shift(6)
    df_normalized['day-7'] = df_normalized['Close'].shift(7)

    #df_normalized['Close'] = df['Close']
    df.dropna(inplace=True)
    df_normalized.dropna(inplace=True)
    df_normalized.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # move 'Close' column to last column
    cols = df_normalized.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df[cols]
    df_normalized = df_normalized[cols]
    
    return df, df_normalized

def plot_data(df, title="Close Price History"):
    plt.figure(figsize=(16,8))
    plt.title(title)
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()