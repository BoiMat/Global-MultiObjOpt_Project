import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
from joblib import cpu_count

class Rule(object):
    
    def __init__(self, function, name, rule_type, arity):
        self.function = function
        self.name = name
        self.rule_type = rule_type
        self.arity = arity
        
    def __call__(self, *args):
        if len(args) != self.arity:
            raise ValueError(f'Incorrect number of arguments for rule: {self.name}. {self.arity} arguments expected but {len(args)} were given.')
        return self.function(*args)
        
    def __str__(self, *args):
        rule = self.name + '(' + ', '.join([str(arg) for arg in args]) + ')' + ' -> ' + self.rule_type
        return rule
    
    def __repr__(self):
        return self.__str__()
    
    def is_buyrule(self):
        if self.rule_type == 'buy':
            return True
        return False
    
    def is_sellrule(self):
        if self.rule_type == 'sell':
            return True
        return False
    
    def print(self):
        print(self.__str__())
    
def cross_below(arg1, arg2):
    if arg1[-1] < arg2[-1] and arg1[-2] > arg2[-2]:
        return True
    return False
    
def cross_above(arg1, arg2):
    if arg1[-1] > arg2[-1] and arg1[-2] < arg2[-2]:
        return True
    return False
    
def grt(arg1, arg2):
    if arg1[-1] > arg2[-1]:
        return True
    return False

def lss(arg1, arg2):
    if arg1[-1] < arg2[-1]:
        return True
    return False

def grt_mul(arg1, arg2, arg3):
    if arg1[-1] > arg2[-1]*arg3[-1]:
        return True
    return False

def lss_mul(arg1, arg2, arg3):
    if arg1[-1] < arg2[-1]*arg3[-1]:
        return True
    return False

def slope_grt(arg1):
    if arg1[-1] - arg1[-2] > 0:
        return True
    return False

def slope_lss(arg1):
    if arg1[-1] - arg1[-2] < 0:
        return True
    return False

def mean_grt(arg1, arg2):
    args = [arg1[-i] for i in range(1,5)]
    if np.mean(args) > arg2[-1]:
        return True
    return False

def mean_lss(arg1, arg2):
    args = [arg1[-i] for i in range(1,5)]
    if np.mean(args) < arg2[-1]:
        return True
    return False

def volatility_grt(arg1, arg2):
    args = [arg1[-i] for i in range(1,5)]
    if np.std(args) > arg2[-1]:
        return True
    return False

def volatility_lss(arg1, arg2):
    args = [arg1[-i] for i in range(1,5)]
    if np.std(args) < arg2[-1]:
        return True
    return False

def composed_and(rule1, rule2):
    if rule1 and rule2:
        return True
    return False

def composed_or(rule1, rule2):
    if rule1 or rule2:
        return True
    return False

cross_below_buy = Rule(cross_below, 'cross_above_buy', 'buy', 2)
cross_below_sell = Rule(cross_below, 'cross_above_sell', 'sell', 2)
cross_above_buy = Rule(cross_above, 'cross_below_buy', 'buy', 2)
cross_above_sell = Rule(cross_above, 'cross_below_sell', 'sell', 2)
grt_buy = Rule(grt, 'grt_buy', 'buy', 2)
grt_sell = Rule(grt, 'grt_sell', 'sell', 2)
lss_buy = Rule(lss, 'lss_buy', 'buy', 2)
lss_sell = Rule(lss, 'lss_sell', 'sell', 2)
grt_mul_buy = Rule(grt_mul, 'grt_mul_buy', 'buy', 3)
grt_mul_sell = Rule(grt_mul, 'grt_mul_sell', 'sell', 3)
lss_mul_buy = Rule(lss_mul, 'lss_mul_buy', 'buy', 3)
lss_mul_sell = Rule(lss_mul, 'lss_mul_sell', 'sell', 3)
slope_grt_buy = Rule(slope_grt, 'slope_grt_buy', 'buy', 1)
slope_grt_sell = Rule(slope_grt, 'slope_grt_sell', 'sell', 1)
slope_lss_buy = Rule(slope_lss, 'slope_lss_buy', 'buy', 1)
slope_lss_sell = Rule(slope_lss, 'slope_lss_sell', 'sell', 1)
mean_grt_buy = Rule(mean_grt, 'mean_grt_buy', 'buy', 2)
mean_grt_sell = Rule(mean_grt, 'mean_grt_sell', 'sell', 2)
mean_lss_buy = Rule(mean_lss, 'mean_lss_buy', 'buy', 2)
mean_lss_sell = Rule(mean_lss, 'mean_lss_sell', 'sell', 2)
volatility_grt_buy = Rule(volatility_grt, 'volatility_grt_buy', 'buy', 2)
volatility_grt_sell = Rule(volatility_grt, 'volatility_grt_sell', 'sell', 2)
volatility_lss_buy = Rule(volatility_lss, 'volatility_lss_buy', 'buy', 2)
volatility_lss_sell = Rule(volatility_lss, 'volatility_lss_sell', 'sell', 2)

composed_and_buy = Rule(composed_and, 'composed_and_buy', 'buy', 2)
composed_and_sell = Rule(composed_and, 'composed_and_sell', 'sell', 2)
composed_or_buy = Rule(composed_or, 'composed_or_buy', 'buy', 2)
composed_or_sell = Rule(composed_or, 'composed_or_sell', 'sell', 2)

def get_buy_rules():
    return [cross_below_buy, cross_above_buy, grt_buy, lss_buy, grt_mul_buy, lss_mul_buy, slope_grt_buy, slope_lss_buy, mean_grt_buy, mean_lss_buy, volatility_grt_buy, volatility_lss_buy]

def get_sell_rules():
    return [cross_below_sell, cross_above_sell, grt_sell, lss_sell, grt_mul_sell, lss_mul_sell, slope_grt_sell, slope_lss_sell, mean_grt_sell, mean_lss_sell, volatility_grt_sell, volatility_lss_sell]

def get_composed_rules():
    return [composed_and_buy, composed_or_buy, composed_and_sell, composed_or_sell]

def default_rules_set():
    buy_rules = get_buy_rules()
    sell_rules = get_sell_rules()
    composed_rules = get_composed_rules()
    
    return [buy_rules, sell_rules, composed_rules]
    
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
    df['EMA9'] = EMAIndicator(close=df["Close"], window=11, fillna=True).ema_indicator()
    df["EMA21"] = EMAIndicator(close=df["Close"], window=21, fillna=True).ema_indicator()
    df["EMA130"] = EMAIndicator(close=df["Close"], window=130, fillna=True).ema_indicator()
    df["EMA200"] = EMAIndicator(close=df["Close"], window=200, fillna=True).ema_indicator()
    
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
