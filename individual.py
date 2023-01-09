import numpy as np
from utils import _Function
from copy import copy

class Tree(object):

    def __init__(self, function_set, arities, init_depth, init_method, n_features, const_range, p_point_replace, parsimony_coefficient, init_investment, random_state, feature_names=None, program=None):
        
        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1]+1)
        self.n_features = n_features
        self.parsimony_coefficient = parsimony_coefficient
        self.const_range = const_range
        self.p_point_replace = p_point_replace
        self.program = program
        self.feature_names = feature_names
        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        
        self.investment = init_investment
        self.buy_price = None
        self.in_the_market = False
        self.buy_ops = []
        self.sell_ops = []
        
        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.generate_tree(random_state)
        
    def generate_tree(self, random_state):
        
        method = ('full' if random_state.randint(2) else 'grow')
        
        max_depth = random_state.randint(*self.init_depth)
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        
        program = [function]
        terminal_stack = [function.arity]
        
        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)

            if depth < max_depth and (method == 'full' or choice <= len(self.function_set)):
                
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)

            else:
                
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                
                program.append(terminal)
                
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        
        return None
    
    def validate_program(self):
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]
    
    def __str__(self):
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output
    
    def export_graphviz(self, fade_nodes=None):
        
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None
    
    def __repr__(self):
        return self.__str__()
    
    def execute(self, X):

        node = self.program[0]
        if isinstance(node, float):
            return node
        if isinstance(node, int):
            return X[node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [t if isinstance(t, float)
                             else X[t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return float(intermediate_result)

        return None
    
    def evaluate(self, X):
        
        res = np.tanh(self.execute(X))
        
        if res > 0.5:
            return 1
        if res < -0.5:
            return -1
        return 0
    
    def reproduce(self):
        return copy(self.program)
    
    def get_subtree(self, random_state, program=None):
        
        if program is None:
            program = self.program

        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end
    
    def crossover(self, donor, random_state):

        start, end = self.get_subtree(random_state)
        removed = range(start, end)

        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))

        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed
        
    def subtree_mutation(self, random_state):
        
        chicken = self.generate_tree(random_state)
        return self.crossover(chicken, random_state)
    
    def hoist_mutation(self, random_state):
        
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        
        return self.program[:start] + hoist + self.program[end:], removed
    
    def point_mutation(self, random_state):

        program = copy(self.program)

        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)

                program[node] = terminal

        return program, list(mutate)
    
    def buy(self, current_price):
        self.buy_price = current_price
        self.in_the_market = True
    
    def sell(self, current_price):
        self.in_the_market = False
        self.investment = self.investment / self.buy_price * current_price
        
    def buy_sell_op(self, X, current_price, i):
    
        res = self.evaluate(X)

        if res == 1:
            if self.in_the_market == False:
                self.buy_ops.append(i)
                self.buy(current_price)
        if res == -1:
            if self.in_the_market:
                self.sell_ops.append(i)
                self.sell(current_price)
        
    def raw_fitness(self, init_investment, X):
        
        self.investment = init_investment
        
        for i in range(X.shape[0]):
            if self.investment <= 0:
                break
            self.buy_sell_op(X[i,:-1], X[i,-1], i)
        
        # if we are in the market at the end of the time series, sell    
        if self.in_the_market:
            self.sell_ops.append(i)
            self.sell(X[-1,-1])
            self.in_the_market = False

        # if we lost some money, return 0
        if self.investment < init_investment:
            return 0
        
        return self.investment - init_investment
    
    def fitness(self, raw_fitness, parsimony_coefficient=None):
        
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program)
        return raw_fitness - penalty
    
    def _length(self):
        return len(self.program)
    
    length_ = property(_length)
