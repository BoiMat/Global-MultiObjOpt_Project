from individual import Tree
from genetic import SymbolicMaximizer
from data import *
import graphviz
import pickle


def main(dataset_func = BTC_1d_Dataset, load=False, save=False):
    
    df, df_normalized = dataset_func()

    features = df_normalized.columns[:-1]

    dataset = np.array(df_normalized[:-200])
    test = np.array(df_normalized[-200:])
    
    function_set = default_function_set()
    
    if load:
        with open('gp_model.pkl', 'rb') as f:
            gp = pickle.load(f)
    else:
        gp = SymbolicMaximizer(population_size=300, generations=300,
                            tournament_size=20, init_depth=(2, 6), 
                            function_set=function_set,
                            parsimony_coefficient=0.01, p_hoist_mutation=0.05, 
                            feature_names=features, 
                            n_jobs=-1, verbose=1, random_state=42)

    gp.fit(dataset, 100)
    
    if save:
        with open('gp_SP500_300p_300g.pkl', 'wb') as f:
            pickle.dump(gp, f)

    # graph = gp._program.export_graphviz()
    # graph = graphviz.Source(graph)
    # graph.render('tree')

    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df_normalized['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    # plot a buy signal
    plt.scatter(df_normalized.index[gp._program.buy_ops], df_normalized['Close'][gp._program.buy_ops], color='green', label='Buy', marker='^', alpha=1)
    # plot a sell signal
    plt.scatter(df_normalized.index[gp._program.sell_ops], df_normalized['Close'][gp._program.sell_ops], color='red', label='Sell', marker='v', alpha=1)
    plt.savefig('Close_SP500_300p_300g.png')
    #plt.show()
    
    
if __name__ == '__main__':
    main(dataset_func = SP500_1d_Dataset, save=True)
