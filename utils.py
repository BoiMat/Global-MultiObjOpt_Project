import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
from joblib import cpu_count

class _Function(object):

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
def _add3(x1, x2, x3):
    return x1 + x2 + x3

def _add4(x1, x2, x3, x4):
    return x1 + x2 + x3 + x4

def _sub3(x1, x2, x3):
    return x1 - x2 - x3

def _sub4(x1, x2, x3, x4):
    return x1 - x2 - x3 - x4

def _mul3(x1, x2, x3):
    return x1 * x2 * x3

def _mul4(x1, x2, x3, x4):
    return x1 * x2 * x3 * x4

def _logicalIF(x1, x2, x3):
    return np.where(x1 > 0, x2, x3)

def _logicalIFGT(x1, x2, x3, x4):
    return np.where(x1 > x2, x3, x4)

def default_function_set():
    add2 = _Function(function=np.add, name='add', arity=2)
    sub2 = _Function(function=np.subtract, name='sub', arity=2)
    mul2 = _Function(function=np.multiply, name='mul', arity=2)

    add3 = _Function(_add3, 'add3', 3)
    sub3 = _Function(_sub3, 'sub3', 3)
    mul3 = _Function(_mul3, 'mul3', 3)
    add4 = _Function(_add4, 'add4', 4)
    sub4 = _Function(_sub4, 'sub4', 4)
    mul4 = _Function(_mul4, 'mul4', 4)

    IF = _Function(_logicalIF, 'IF', 3)
    IFGT = _Function(_logicalIFGT, 'IFGT', 4)
    
    return [add2, sub2, mul2, add3, sub3, mul3, add4, sub4, mul4, IF, IFGT]
    
def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
    
def _get_n_jobs(n_jobs):

    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs

def _partition_estimators(n_estimators, n_jobs):
    
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
    
def AddIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    
    # Add Bollinger Bands indicator
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
    df['psar'] = indicator_psar.psar()

    # Add Moving Average Convergence Divergence (MACD) indicator
    df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True) # mazas

    # Add Relative Strength Index (RSI) indicator
    df["RSI"] = rsi(close=df["Close"], window=14, fillna=True) # mazas
    
    # Add Exponential Moving Average (EMA) indicator
    df['EMA11'] = EMAIndicator(close=df["Close"], window=11, fillna=True).ema_indicator()
    df["EMA21"] = EMAIndicator(close=df["Close"], window=21, fillna=True).ema_indicator()
    df["EMA50"] = EMAIndicator(close=df["Close"], window=50, fillna=True).ema_indicator()
    df["EMA200"] = EMAIndicator(close=df["Close"], window=200, fillna=True).ema_indicator()
    
    # Add McClellan Oscillator indicator
    # df["MOM"] = mom(close=df["Close"], window=10, fillna=True)
    
    return df

def DropCorrelatedFeatures(df, threshold, plot):
    df_copy = df.copy()

    # Remove OHCL columns
    df_drop = df_copy.drop(["Date", "Open", "High", "Low", "Close", "Volume"], axis=1)

    # Calculate Pierson correlation
    df_corr = df_drop.corr()

    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[0]):
            if df_corr.iloc[i,j] >= threshold or df_corr.iloc[i,j] <= -threshold:
                if columns[j]:
                    columns[j] = False
                    
    selected_columns = df_drop.columns[columns]

    df_dropped = df_drop[selected_columns]

    if plot:
        # Plot Heatmap Correlation
        fig = plt.figure(figsize=(8,8))
        ax = sns.heatmap(df_dropped.corr(), annot=True, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        fig.tight_layout()
        plt.show()
    
    return df_dropped