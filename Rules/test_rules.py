from individual_rules import Rules_bot
from genetic_rules import SymbolicMaximizer
from data_rules import *
from utils_rules import *
import graphviz
import pickle
import time


def main(dataset_func = BTC_1d_Dataset, population=200, generations=200, zscore=False, verbose=0, load=False, save=False):
    
    pop = population
    gen = generations
    
    name = f'BTC_{pop}p_{gen}g_rules'
    path = 'models/' + name + '.pkl'

    df, df_normalized = dataset_func(zscore=zscore)

    prices = np.array(df['Close'][:-200])
    prices_test = np.array(df['Close'][-200:])

    df.drop(['Close'], axis=1, inplace=True)
    
    dataset = np.array(df_normalized[:-200])
    test = np.array(df_normalized[-200:])
    
    indicators = df.columns
    dataset = np.array(df[:-200])
    test = np.array(df[-200:])
    
    rules = default_rules_set()
    
    if load:
        with open(path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = SymbolicMaximizer(population_size=pop, generations=gen,
                               tournament_size=20, rules_set=rules, 
                               indicators_set=indicators, 
                               n_jobs=-1, verbose=verbose, 
                               random_state=1, elitism=True)

    gp.fit(dataset, prices, 100)
    
    if save:
        with open(path, 'wb') as f:
            pickle.dump(gp, f)

    plt.figure(figsize=(16,8))
    plt.plot(prices)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    # plot a buy signal
    plt.scatter(df.index[gp._program.buy_ops], prices[gp._program.buy_ops], color='green', label='Buy', marker='^', alpha=1)
    plt.scatter(df.index[gp._program.sell_ops], prices[gp._program.sell_ops], color='red', label='Sell', marker='v', alpha=1)
    plt.show()
    if save:
        plt.savefig('images/' + name + '.png')
    
    
if __name__ == '__main__':
    main(dataset_func = BTC_1d_Dataset, population=200, generations=200, zscore=False, verbose=0, load=False, save=True)
