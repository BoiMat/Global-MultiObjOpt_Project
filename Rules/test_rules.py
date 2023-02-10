from individual_rules import Rules_bot
from genetic_rules import SymbolicMaximizer
from data_rules import *
from utils_rules import *
import graphviz
import pickle
import time


def main(dataset_func = BTC_1d_Dataset, population=200, generations=200, zscore=False, elitism=False, verbose=0, load=False, save=False):
    
    pop = population
    gen = generations
    
    name = f'BTC_{pop}p_{gen}g_rules_'
    name += 'zscore_' if zscore else 'close_'
    name += 'elitism_' if elitism else ''
    path = 'models/' + name + '.pkl'

    df, _ = dataset_func(zscore=zscore)

    prices = np.array(df['Entry_Price'][:-365])
    prices_test = np.array(df['Entry_Price'][-365:])

    df.drop(['Entry_Price'], axis=1, inplace=True)
    
    dataset = np.array(df[:-365])
    test = np.array(df[-365:])
    
    indicators = df.columns
    
    rules = default_rules_set()
    
    if load:
        with open(path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = SymbolicMaximizer(population_size=pop, generations=gen,
                               tournament_size=20, max_num_rules=15, 
			                   rules_set=rules, indicators_set=indicators, 
                               n_jobs=-1, verbose=verbose, 
                               random_state=1, elitism=elitism)

    gp.fit(dataset, prices, 100)
    
    if save:
        with open(path, 'wb') as f:
            pickle.dump(gp, f)
            
    plot_results(prices, gp._program.buy_ops, gp._program.sell_ops, title=name, save=True, name=name)
    
    length_buy_ops = len(gp._program.buy_ops)
    gp.predict(test, prices_test, 100, verbose=True)
    
    plot_results(prices_test, gp._program.buy_ops[length_buy_ops:], gp._program.sell_ops[length_buy_ops:], title=name, save=True, name=name+'_test')
    
    
if __name__ == '__main__':
    main(dataset_func = BTC_1d_Dataset, population=500, generations=500, zscore=True, elitism=False, verbose=0, load=False, save=True)
