from genetic_multiobj import SymbolicMaximizer
from data import *
import pickle

def main(dataset_func = BTC_1d_Dataset, population=200, generations=200, zscore=False, elitism=False, verbose=0, load=False, save=False, warm_start_gen=None):
    
    pop = population
    gen = generations
    
    name = f'BTC_{pop}p_{gen}g_multiobj'
    name += '_zscore' if zscore else '_close'
    name += '_elitism' if elitism else ''
    path = 'models/' + name + '.pkl'

    _, df_normalized = dataset_func(zscore=zscore)

    features = df_normalized.columns[:-1]

    dataset = np.array(df_normalized[:-365])
    test = np.array(df_normalized[-365:])
    
    function_set = default_function_set()
    
    if load == False:
        gp = SymbolicMaximizer(population_size=pop, generations=gen,
                            tournament_size=20, init_depth=(2, 6), 
                            function_set=function_set,
                            parsimony_coefficient=0.5, p_hoist_mutation=0.05, 
                            feature_names=features, elitism=elitism,
                            n_jobs=-1, verbose=verbose, random_state=1)
        
        gp.fit(dataset, 1)
         
    else:
        with open(path, 'rb') as f:
            gp = pickle.load(f)
            
        if warm_start_gen is not None:
            gp.set_params(generations = gen, warm_start=True)
            gp.fit(dataset, 1)
	    path = path.replace(f'{gen}g', f'{gen+warm_start_gen}g')
        
    if save:
        with open(path, 'wb') as f:
            pickle.dump(gp, f)

    plot_results(dataset, gp._program.buy_ops, gp._program.sell_ops, title=name, save=True, name=name)
    
    length_buy_ops = len(gp._program.buy_ops)
    gp.predict(test, 1, verbose=True)
    
    plot_results(test, gp._program.buy_ops[length_buy_ops:], gp._program.sell_ops[length_buy_ops:], title=name, save=True, name=name+'_test')
    
    
if __name__ == '__main__':
    main(dataset_func = BTC_1d_Dataset, population=30, generations=20, zscore=False, elitism=False, verbose=0, load=False, save=True, warm_start_gen=None)
