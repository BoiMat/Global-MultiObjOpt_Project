from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler

def BTC_1d_Dataset(zscore = False):
    
    df = pd.read_csv('Binance_BTCUSDT_d.csv')
    df.drop(['symbol', 'unix', 'Volume USDT', 'tradecount'], axis=1, inplace=True)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.sort_values('Date')
    df = AddIndicators(df)
    
    yhat = 'Close' if zscore == False else 'zscore'
    
    if zscore:
       zscore_col = (df['Close'] - df['Close'].rolling(200).mean()) / df['Close'].rolling(200).std()
    
    df.drop(['Open', 'High', 'Low', 'Date'], axis=1, inplace=True)
    
    # # normalize the dataset between -1 and 1
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    df_normalized = df.copy()
    for column in df_normalized.columns:
       df_normalized[column] = (df_normalized[column] - df_normalized[column].mean()) / df_normalized[column].std()
    
    if zscore:        
       df['zscore'] = zscore_col
       df_normalized['zscore'] = zscore_col
       
    temp1 = df.pop(yhat)
    temp2 = df_normalized.pop(yhat)
    df[yhat] = temp1
    df_normalized[yhat] = temp2
    
    df[f'{yhat}-1'] = df[yhat].shift(1)
    df[f'{yhat}-2'] = df[yhat].shift(2)
    df[f'{yhat}-3'] = df[yhat].shift(3)
    df[f'{yhat}-4'] = df[yhat].shift(4)
    df[f'{yhat}-5'] = df[yhat].shift(5)
    df[f'{yhat}-6'] = df[yhat].shift(6)
    
    df_normalized[f'{yhat}-1'] = df_normalized[yhat].shift(1)
    df_normalized[f'{yhat}-2'] = df_normalized[yhat].shift(2)
    df_normalized[f'{yhat}-3'] = df_normalized[yhat].shift(3)
    df_normalized[f'{yhat}-4'] = df_normalized[yhat].shift(4)
    df_normalized[f'{yhat}-5'] = df_normalized[yhat].shift(5)
    df_normalized[f'{yhat}-6'] = df_normalized[yhat].shift(6)
    
    df['Entry_Price'] = df['Close'].shift(-1)
    df_normalized['Entry_Price'] = df['Entry_Price']

    #df_normalized['Close'] = df['Close']
    df = df.replace(0, 0.000001)
    df_normalized = df_normalized.replace(0, 0.000001)
    df.dropna(inplace=True)
    df_normalized.dropna(inplace=True)
    df_normalized.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # temp = df.pop('Close')
    # df_normalized.pop('Close')
    # df = df.assign(Close=temp)
    # df_normalized = df_normalized.assign(Close=temp)
    
    if zscore:
        df.drop(['Close'], axis=1, inplace=True)
        df_normalized.drop(['Close'], axis=1, inplace=True)
    
    return df, df_normalized

def SP500_1d_Dataset(zscore = False):
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
    
    if zscore:
        zscore_col = (df['Close'] - df['Close'].rolling(200).mean()) / df['Close'].rolling(200).std()

    yhat = 'Close' if zscore == False else 'zscore'
    # # normalize the dataset between -1 and 1
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
    # df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
    df_normalized = df.copy()
    for column in df_normalized.columns:
       df_normalized[column] = (df_normalized[column] - df_normalized[column].mean()) / df_normalized[column].std()
    
    if zscore:        
       df['zscore'] = zscore_col
       df_normalized['zscore'] = zscore_col
       
    temp1 = df.pop(yhat)
    temp2 = df_normalized.pop(yhat)
    df[yhat] = temp1
    df_normalized[yhat] = temp2
    
    df[f'{yhat}-1'] = df[yhat].shift(1)
    df[f'{yhat}-2'] = df[yhat].shift(2)
    df[f'{yhat}-3'] = df[yhat].shift(3)
    df[f'{yhat}-4'] = df[yhat].shift(4)
    df[f'{yhat}-5'] = df[yhat].shift(5)
    df[f'{yhat}-6'] = df[yhat].shift(6)
    
    df_normalized[f'{yhat}-1'] = df_normalized[yhat].shift(1)
    df_normalized[f'{yhat}-2'] = df_normalized[yhat].shift(2)
    df_normalized[f'{yhat}-3'] = df_normalized[yhat].shift(3)
    df_normalized[f'{yhat}-4'] = df_normalized[yhat].shift(4)
    df_normalized[f'{yhat}-5'] = df_normalized[yhat].shift(5)
    df_normalized[f'{yhat}-6'] = df_normalized[yhat].shift(6)
    
    df['Entry_Price'] = df['Close'].shift(-1)
    df_normalized['Entry_Price'] = df['Entry_Price']

    #df_normalized['Close'] = df['Close']
    df = df.replace(0, 0.000001)
    df_normalized = df_normalized.replace(0, 0.000001)
    df.dropna(inplace=True)
    df_normalized.dropna(inplace=True)
    df_normalized.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # temp = df.pop('Close')
    # df_normalized.pop('Close')
    # df = df.assign(Close=temp)
    # df_normalized = df_normalized.assign(Close=temp)
    
    if zscore:
        df.drop(['Close'], axis=1, inplace=True)
        df_normalized.drop(['Close'], axis=1, inplace=True)
        
    return df, df_normalized

def plot_data(df, title="Close Price History"):
    plt.figure(figsize=(16,8))
    plt.title(title)
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()
