from individual import Tree
from genetic import SymbolicMaximizer
from data import *
import graphviz
import pickle
import time


def main(dataset_func = BTC_1d_Dataset, population=200, generations=200, zscore=False, verbose=0, load=False, save=False):
    
    pop = population
    gen = generations
    
    name = f'BTC_{pop}p_{gen}g_elitism'
    path = 'models/' + name + '.pkl'

    df, df_normalized = dataset_func(zscore=zscore)

    features = df_normalized.columns[:-1]

    dataset = np.array(df_normalized[:-400])
    test = np.array(df_normalized[-400:])
    
    function_set = default_function_set()
    
    if load:
        with open(path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = SymbolicMaximizer(population_size=pop, generations=gen,
                            tournament_size=20, init_depth=(2, 6), 
                            function_set=function_set,
                            parsimony_coefficient=0.01, p_hoist_mutation=0.05, 
                            feature_names=features, elitism=True,
                            n_jobs=-1, verbose=verbose, random_state=1)

    gp.fit(dataset, 1)
    
    if save:
        with open(path, 'wb') as f:
            pickle.dump(gp, f)

    plt.figure(figsize=(16,8))
    plt.title(name)
    plt.plot(df_normalized.iloc[:,-1])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    # plot a buy signal
    plt.scatter(df_normalized.index[gp._program.buy_ops], df_normalized.iloc[:,-1][gp._program.buy_ops], color='green', label='Buy', marker='^', alpha=1)
    # plot a sell signal
    plt.scatter(df_normalized.index[gp._program.sell_ops], df_normalized.iloc[:,-1][gp._program.sell_ops], color='red', label='Sell', marker='v', alpha=1)
    if save:
        plt.savefig('images/' + name + '.png')
    
    
if __name__ == '__main__':
    main(dataset_func = BTC_1d_Dataset, population=500, generations=200, zscore=False, verbose=0, load=False, save=True)
