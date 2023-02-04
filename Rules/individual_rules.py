import numpy as np
from utils_rules import Rule
from copy import copy

class Rules_bot(object):

    def __init__(self, max_num_rules, rules, indicators, random_state, init_investment=100, program=None):
        
        self.program = program
        self.rules = rules
        self.indicators = indicators
        
        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        
        self.buy_price = 0
        self.in_the_market = False
        self.buy_ops = []
        self.sell_ops = []
        self.investment = init_investment
        
        
        if self.program is not None:
            if not self.validate_program(rules):
                raise ValueError(f'The supplied program: {self.program}, is incomplete.')
        else:
            # Create a naive random program
            self.program = self.generate_program(max_num_rules, rules, indicators, random_state)

    def generate_new_rule(self, random_state, rules, indicators, specific_rule_type=None):
      
      if specific_rule_type is None:
        rule_type = random_state.randint(0, len(rules))
      else:
        rule_type = specific_rule_type
      
      rule_index = random_state.randint(0, len(rules[rule_type]))
      arity = rules[rule_type][rule_index].arity
        
      if rule_type == 2:
        rule1 = self.generate_new_rule(random_state, rules[0:2], indicators)
        rule2 = self.generate_new_rule(random_state, rules[0:2], indicators)
        rule = [rule_type, rule_index, rule1, rule2]
        return rule  
    
      indicator_indeces = random_state.randint(0, len(indicators), arity)
      rule = [rule_type, rule_index, indicator_indeces]
      return rule

    def generate_program(self, max_num_rules, rules, indicators, random_state):
      
      program = []
      
      number_of_rules = random_state.randint(2, max_num_rules)
      for _ in range(number_of_rules-1):
        rule = self.generate_new_rule(random_state, rules, indicators)
        program.append(rule)
        
      if self.validate_program(rules, program) == False:
        rule_type = program[-1][0]
        rule_index = program[-1][1]
        if rules[rule_type][rule_index].is_buyrule():
          rule = self.generate_new_rule(random_state, rules, indicators, 1)
        else:
          rule = self.generate_new_rule(random_state, rules, indicators, 0)
        program.append(rule)
      else:
        rule = self.generate_new_rule(random_state, rules, indicators)
        program.append(rule)
        
      return program
        
    def validate_program(self, rules, program = None):
      
      if program is None:
        program = self.program
      
      #throw error if program has less than 2 rules
      if len(program) < 2:
        return False
      
      #throw error if program does not have a buy and sell rule
      buy_rule = False
      sell_rule = False
      for rule in program:
        if rules[rule[0]][rule[1]].is_buyrule():
          buy_rule = True
        if rules[rule[0]][rule[1]].is_sellrule():
          sell_rule = True
      if not buy_rule or not sell_rule:
        return False
      
      return True
      
    def __str__(self) -> str:
      return self.program
         
    def __repr__(self):
      return self.__str__()
    
    def reproduce(self):
      return copy(self.program)
      
    def crossover(self, donor, random_state):

      self_index = random_state.randint(0, len(self.program))
      removed = self.program[self_index]
      removed_type = self.rules[removed[0]][removed[1]].is_buyrule()

      for i in range(len(donor)):
        if self.rules[donor[i][0]][donor[i][1]].is_buyrule() == removed_type:
          donor_index = i
          break
      
      donor_removed = donor[donor_index]

      self.program[self_index] = donor_removed
      #donor[donor_index] = removed
      
      return self.program, removed, donor_removed
    
    def rule_mutation(self, random_state):
      
      rule_index = random_state.randint(0, len(self.program))
      removed = self.program[rule_index]
      rule_type = removed[0]
      
      if rule_type == 2:
        rule_type = 0 if self.rules[rule_type][removed[1]].is_buyrule() else 1
      
      chicken = self.generate_new_rule(random_state, self.rules, self.indicators, rule_type)
      self.program[rule_index] = chicken
      return self.program, removed, chicken
  
    def indicators_mutation(self, random_state):
      
      rule_index = random_state.randint(0, len(self.program))
      rule_type = self.program[rule_index][0]
      
      if rule_type == 2:
        subrule_index = random_state.randint(2, 4)
        arity = self.rules[self.program[rule_index][subrule_index][0]][self.program[rule_index][subrule_index][1]].arity
        indicator_indeces = random_state.randint(0, len(self.indicators), arity)
        removed = self.program[rule_index][subrule_index][2]
        self.program[rule_index][subrule_index][2] = indicator_indeces
      else:
        arity = self.rules[self.program[rule_index][0]][self.program[rule_index][1]].arity
        indicator_indeces = random_state.randint(0, len(self.indicators), arity)
        removed = self.program[rule_index][2]
        self.program[rule_index][2] = indicator_indeces
      
      return self.program, removed
    
    def indicator_mutation(self, random_state):

      rule_index = random_state.randint(0, len(self.program))
      rule_type = self.program[rule_index][0]
      
      if rule_type == 2:
        subrule_index = random_state.randint(2, 4)
        arity = self.rules[self.program[rule_index][subrule_index][0]][self.program[rule_index][subrule_index][1]].arity
        indicator = random_state.randint(0, len(self.indicators))
        indicator_index = random_state.randint(0, arity)
        removed = self.program[rule_index][subrule_index][2][indicator_index]
        self.program[rule_index][subrule_index][2][indicator_index] = indicator
      else:
        arity = self.rules[self.program[rule_index][0]][self.program[rule_index][1]].arity
        indicator = random_state.randint(0, len(self.indicators))
        indicator_index = random_state.randint(0, arity)
        removed = self.program[rule_index][2][indicator_index]
        self.program[rule_index][2][indicator_index] = indicator
        
      return self.program, removed
    
    def evaluate_simple(self, data, rule, rules_list):
      exec_rule = rules_list[rule[0]][rule[1]]
      return exec_rule.function(*data[:, rule[2]].T)
    
    def evaluate_composed(self, data, rule, rules_list):
      result1 = self.evaluate_simple(data, rule[2], rules_list)
      result2 = self.evaluate_simple(data, rule[3], rules_list)
      return rules_list[rule[0]][rule[1]].function(result1, result2)
    
    def evaluate(self, data_window, rules):
      buy = False
      sell = False

      for rule in self.program:
        if rule[0] == 2:
          result = self.evaluate_composed(data_window, rule, rules)
        else:
          result = self.evaluate_simple(data_window, rule, rules)
        
        # print(f'rule: {rules[rule[0]][rule[1]].name}, isbuy: {rules[rule[0]][rule[1]].is_buyrule()}, issell: {rules[rule[0]][rule[1]].is_sellrule()}, result : {result}')
        
        if result:
          if rules[rule[0]][rule[1]].is_buyrule():
            buy = True
          else:
            sell = True
      
      return buy, sell
    
    def buy(self, current_price):
      self.buy_price = current_price
      self.in_the_market = True
    
    def sell(self, current_price):
      self.in_the_market = False
      self.investment = self.investment / self.buy_price * current_price
      
    def buy_sell_op(self, data_window, current_price, i, rules):
    
      buy, sell = self.evaluate(data_window, rules)

      if buy:
        if self.in_the_market == False:
          self.buy_ops.append(i)
          self.buy(current_price)
      if sell:
        if self.in_the_market:
          self.sell_ops.append(i)
          self.sell(current_price - current_price*0.003)
          
    def raw_fitness(self, data, prices, rules, init_investment):
      self.investment = init_investment
      for i in range(4, len(data)):
        data_window = data[i-4:i,:]
        current_price = prices[i]
        self.buy_sell_op(data_window, current_price, i, rules)
      return self.investment
    
    def length(self):
        return len(self.program)