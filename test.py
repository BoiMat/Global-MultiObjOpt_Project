from individual import Tree
from genetic import SymbolicMaximizer
from data import *
import graphviz
import pickle


def main(dataset_func = BTC_1d_Dataset, load=False, save=False):
    
    name = 'zscore_BTC_200p_200g'
    path = 'models/' + name + '.pkl'

    df, df_normalized = dataset_func(zscore=True)

    features = df_normalized.columns[:-1]

    dataset = np.array(df_normalized[:-200])
    test = np.array(df_normalized[-200:])
    
    function_set = default_function_set()
    
    if load:
        with open(path, 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = SymbolicMaximizer(population_size=200, generations=200,
                            tournament_size=20, init_depth=(2, 6), 
                            function_set=function_set,
                            parsimony_coefficient=0.01, p_hoist_mutation=0.05, 
                            feature_names=features, 
                            n_jobs=-1, verbose=0, random_state=42)

    gp.fit(dataset, 100)
    
    if save:
        with open(path, 'wb') as f:
            pickle.dump(gp, f)

    # graph = gp._program.export_graphviz()
    # graph = graphviz.Source(graph)
    # graph.render('tree')

    plt.figure(figsize=(16,8))
    plt.title(name)
    plt.plot(df_normalized.iloc[:,-1])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    # plot a buy signal
    plt.scatter(df_normalized.index[gp._program.buy_ops], df_normalized['Close'][gp._program.buy_ops], color='green', label='Buy', marker='^', alpha=1)
    # plot a sell signal
    plt.scatter(df_normalized.index[gp._program.sell_ops], df_normalized['Close'][gp._program.sell_ops], color='red', label='Sell', marker='v', alpha=1)
    plt.savefig(name + '.png')
    #plt.show()
    
    
if __name__ == '__main__':
    main(save=True)
