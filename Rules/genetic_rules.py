import itertools
from abc import ABCMeta, abstractmethod
from time import time
from joblib import Parallel, delayed
from utils_rules import Rule, check_random_state, _get_n_jobs, _partition_estimators
from individual_rules import Rules_bot
import numpy as np
from sklearn.base import BaseEstimator

MAX_INT = np.iinfo(np.int32).max

def _parallel_evolve(n_programs, parents, data, prices, init_investment, seeds, params, elitism):

    tournament_size = params['tournament_size']
    rules_set = params['rules_set']
    max_num_rules = params['max_num_rules']
    method_probs = params['method_probs']
    indicators_set = params['indicators_set']

    def tournament_selection():
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].raw_fitness_ for p in contenders]
        parent_index = contenders[np.argmax(fitness)]
        return parents[parent_index], parent_index

    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = tournament_selection()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = tournament_selection()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # rule_mutation
                program, removed, _ = parent.rule_mutation(random_state)
                genome = {'method': 'Rule Mutation',
                          'parent_idx': parent_index,
                          'parent_node': removed}
            elif method < method_probs[2]:
                # indicators_mutation
                program, removed = parent.indicators_mutation(random_state)
                genome = {'method': 'Indicators Mutation',
                          'parent_idx': parent_index,
                          'parent_node': removed}
            elif method < method_probs[3]:
                # indicator_mutation
                program, mutated = parent.indicator_mutation(random_state)
                genome = {'method': 'Indicator Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}
                
        program = Rules_bot( max_num_rules = max_num_rules,
                             rules=rules_set,
                             indicators=indicators_set,
                             random_state=random_state,
                             init_investment=init_investment,
                             program=program )

        program.parents = genome      
        
        program.raw_fitness_ = program.raw_fitness(data, prices, init_investment)

        programs.append(program)
    
    if elitism & (parents is not None):    
        # add the best 20 programs to the next generation
        hall_of_fame = sorted(parents, key=lambda x: x.raw_fitness_, reverse=True)[:20]
        for program in hall_of_fame:
            programs.append(program)

    return programs

class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 const_range=(-1., 1.),
                 max_num_rules=10,
                 #init_method='half and half',
                 rules_set=None,
                 #parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_rule_mutation=0.01,
                 p_indicators_mutation=0.01,
                 p_indicator_mutation=0.01,
                 #p_point_replace=0.05,
                 indicators_set=None,
                 elitism=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.const_range = const_range
        self.max_num_rules = max_num_rules
        # self.init_method = init_method
        self.rules_set = rules_set
        #self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_rule_mutation = p_rule_mutation
        self.p_indicators_mutation = p_indicators_mutation
        self.p_indicator_mutation = p_indicator_mutation
        self.elitism = elitism
        self.indicators_set = indicators_set
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _verbose_reporter(self, run_details=None):

        if run_details is None:
            print('    |{:^25}|{:^25}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 25 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     remaining_time))

    def fit(self, data, prices, init_investment):
        
        random_state = check_random_state(self.random_state)
        
        self._rules_set = []
        for i in range(len(self.rules_set)):
            rule_list = []
            for rule in self.rules_set[i]:
                if isinstance(rule, Rule):
                    rule_list.append(rule)
            self._rules_set.append(rule_list)
        
        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for i in range(len(self._rules_set)):
            for rule in self._rules_set[i]:
                arity = rule.arity
                self._arities[arity] = self._arities.get(arity, [])
                self._arities[arity].append(rule)

        self._method_probs = np.array([self.p_crossover,
                                       self.p_rule_mutation,
                                       self.p_indicators_mutation,
                                       self.p_indicator_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        params = self.get_params()

        params['rules_set'] = self._rules_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs
        params['max_num_rules'] = self.max_num_rules
        
        if not hasattr(self, '_programs'):
            self._programs = []
            self.run_details_ = {'generation': [],
                                'average_length': [],
                                'average_fitness': [],
                                'best_length': [],
                                'best_fitness': [],
                                'generation_time': []}

        prior_generations = len(self._programs)

        if self.verbose:
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)
            
            population = Parallel(n_jobs=n_jobs, verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i], parents, data, prices, init_investment, seeds[starts[i]:starts[i+1]], params, self.elitism) 
                for i in range(n_jobs))
            
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length() for program in population]

            # parsimony_coefficient = None
            # if self.parsimony_coefficient == 'auto':
            #     parsimony_coefficient = (np.cov(length, fitness)[1, 0] / np.var(length))
            # for program in population:
            #     program.fitness_ = program.fitness(program.raw_fitness_, parsimony_coefficient)

            self._programs.append(population)

            if gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            best_program = population[np.argmax(fitness)]
            self._program = best_program

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length())
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

                
            # self._program = self._programs[-1][np.argmax(fitness)]

        return self
    
class SymbolicMaximizer(BaseSymbolic):

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 const_range=(-1., 1.),
                 max_num_rules=10,
                 #init_method='half and half',
                 rules_set=None,
                 #parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_rule_mutation=0.01,
                 p_indicators_mutation=0.01,
                 p_indicator_mutation=0.01,
                 elitism=False,
                 indicators_set=None,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        
        super(SymbolicMaximizer, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            const_range=const_range,
            max_num_rules=max_num_rules,
            #init_method=init_method,
            rules_set=rules_set,
            #parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_rule_mutation=p_rule_mutation,
            p_indicators_mutation=p_indicators_mutation,
            p_indicator_mutation=p_indicator_mutation,
            elitism=elitism,
            indicators_set=indicators_set,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def _more_tags(self):
        return {'binary_only': True}

    def predict(self, data, rules, prices, init_investment = 100):
 
        self._program.investment = init_investment
        score = self._program.raw_fitness(data, prices, init_investment)
        return score - init_investment