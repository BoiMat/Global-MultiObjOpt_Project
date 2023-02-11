from genetic_multiobj import SymbolicMaximizer
from data import *
import pickle

def main(dataset_func = BTC_1d_Dataset, population=200, generations=200, zscore=False, elitism=False, verbose=0, load=False, save=False):
    
    pop = population
    gen = generations
    
    name = f'BTC_{pop}p_{gen}g'
    name += '_zscore' if zscore else '_close'
    name += '_elitism' if elitism else ''
    path = 'models/' + name + '.pkl'

    _, df_normalized = dataset_func(zscore=zscore)

    features = df_normalized.columns[:-1]

    dataset = np.array(df_normalized[:-365])
    test = np.array(df_normalized[-365:])
    
    function_set = default_function_set()
    
    if load:
        with open(path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = SymbolicMaximizer(population_size=pop, generations=gen,
                            tournament_size=20, init_depth=(2, 6), 
                            function_set=function_set,
                            parsimony_coefficient=0.05, p_hoist_mutation=0.05, 
                            feature_names=features, elitism=elitism,
                            n_jobs=-1, verbose=verbose, random_state=1)

    gp.fit(dataset, 1)
    
    if save:
        with open(path, 'wb') as f:
            pickle.dump(gp, f)

    plot_results(dataset, gp._program.buy_ops, gp._program.sell_ops, title=name, save=True, name=name)
    
    length_buy_ops = len(gp._program.buy_ops)
    gp.predict(test, 1, verbose=True)
    
    plot_results(test, gp._program.buy_ops[length_buy_ops:], gp._program.sell_ops[length_buy_ops:], title=name, save=True, name=name+'_test')
    
    
if __name__ == '__main__':
    main(dataset_func = BTC_1d_Dataset, population=500, generations=200, zscore=False, elitism=False, verbose=0, load=False, save=True)
