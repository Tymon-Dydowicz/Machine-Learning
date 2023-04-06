from typing import *
from matplotlib import pyplot as plt
from string import ascii_uppercase
import itertools
import random

Variable = str
DomainElement = Hashable # anything that can be a key of a dictionary

ProbabilityDistribution = Dict[DomainElement, float]

Assignment = Tuple[DomainElement, ...]
ConditionalProbabilityDistribution = Dict[Assignment, ProbabilityDistribution]
Parents = Tuple[Variable, ...]

class BayesianNet:        
    _unconditional: Dict[Variable, ProbabilityDistribution]
    _conditional: Dict[Variable, Tuple[Parents, ConditionalProbabilityDistribution]]
    _domain: Dict[Variable, Iterable[DomainElement]]
    
    def __init__(self):
        self._unconditional = {}
        self._conditional = {}
        self._domain = {}        
        
    def _check_and_normalize(self, distribution: ProbabilityDistribution) -> ProbabilityDistribution:
        assert len(distribution.keys()) >= 2
        if None in distribution.values():
            rem = sum([f for f in distribution.values() if f is not None])
            assert 0 <= rem < 1
            noneKey = [k for k, v in distribution.items() if v is None]
            assert len(noneKey) == 1
            distribution[noneKey[0]] = 1 - rem
        assert all(0<v<1 for v in distribution.values())
        assert sum(distribution.values()) == 1
        return distribution
        
    def addUnconditionalVariable(self, name: Variable, distribution: ProbabilityDistribution) -> None:        
        assert name not in self._conditional
        assert name not in self._unconditional
        distribution = self._check_and_normalize(distribution)
        self._unconditional[name] = distribution
        self._domain[name] = set(distribution.keys())
    
    def addConditionalVariable(self, name: Variable, parents: Parents, cpt: ConditionalProbabilityDistribution) -> None:        
        assert name not in self._conditional
        assert name not in self._unconditional
        assert isinstance(parents, tuple)
        assert len(parents) > 0
        assert all(len(parents) == len(k) for k in cpt.keys())
        domain = set(next(iter(cpt.values())).keys())
        assert all(v.keys() == domain for v in cpt.values())        
        cpt = {k: self._check_and_normalize(distribution) for k, distribution in cpt.items()}
        self._conditional[name] = (parents, cpt)
        self._domain[name] = domain
    
    def addBooleanUnconditionalVariable(self, name: Variable, pTrue: float) -> None:
        assert 0 < pTrue < 1
        self.addUnconditionalVariable(name, {True: pTrue, False: 1-pTrue})
        
    def addBooleanConditionalVariable(self, name: Variable, parents: Parents, cpt: Dict[Assignment, float]):
        cpt = {k: {True: v, False: 1-v} for k, v in cpt.items()}
        self.addConditionalVariable(name, parents, cpt)
        
    def domain(self, name: Variable) -> Iterable[DomainElement]:
        return self._domain[name]
    
    def variables(self) -> Iterable[Variable]:
        return self._conditional.keys() | self._unconditional.keys()
    
    def parents(self, name: Variable) -> Parents:
        if name in self._conditional:
            return self._conditional[name][0]
        else:
            return []
    
    def p(self, name: Variable, value: DomainElement, condition: Dict[Variable, DomainElement]):
        if name in self._conditional:
            parents, cpt = self._conditional[name]
            assert all(p in condition for p in parents)
            condition = tuple(condition[p] for p in parents)
            dist = cpt[condition]
        else:
            assert name in self._unconditional
            dist = self._unconditional[name]
        return dist[value]

bn = BayesianNet()
bn.addUnconditionalVariable('B', {True: 0.001, False: None})
bn.addBooleanUnconditionalVariable('E', 0.002)

parents = ('B', 'E')
cpd = {(True, True): {True: .95, False: None}, 
       (True, False): {True: .94, False: None}, 
       (False, True): {True: .29, False: None}, 
       (False, False): {True: .001, False: None}}

bn.addConditionalVariable('A', parents, cpd)
bn.addBooleanConditionalVariable('J', ('A',), {(True,): .9, (False,): .05})
bn.addBooleanConditionalVariable('M', ('A',), {(True,): .7, (False,): .01})
bn.variables()
bn.domain('A')
bn.parents('A')
bn.p('A', True, {'B': True, 'E': False})
bn.p('A', True, {'B': True, 'E': False, 'M': True, 'J': True})
bn.p('B', True, {'E': False})
bn.p('B', True, None)

def verify_topological_order(bn: BayesianNet, ordering: List[Variable]):
    for i, var in enumerate(ordering):
        assert all(p in ordering[:i] for p in bn.parents(var))

verify_topological_order(bn, ['B', 'E', 'A', 'J', 'M'])

def topological_sort(bn: BayesianNet) -> List[Variable]:
    ...
    def dfs(node, memory):
        if node not in memory:
            memory.append(node)
        for parent in bn.parents(node):
            if parent not in memory:
                dfs(parent, memory)
        return 1

    parents = []

    for node in bn.variables():
        for parent in bn.parents(node):
            if parent not in parents:
                parents.append(parent)

    sortedVariables = [node for node in (set(bn.variables()) - set(parents)) ]       
    for node in sortedVariables:
        dfs(node, sortedVariables)
    
    return sortedVariables[::-1]

verify_topological_order(bn, topological_sort(bn))
topological_sort(bn)

from copy import deepcopy
def enumeration_ask(X: Variable, e: Assignment, bn: BayesianNet) -> ProbabilityDistribution:

    def enumerate_all(vars, e, bn: BayesianNet, sum = 0):
        if len(vars) == 0:
            return 1.0
        Y = vars.pop(0)
        eCopy = deepcopy(e)

        condition = {}
        for parent in bn.parents(Y):
            if parent in eCopy.keys():
                condition[parent] = eCopy[parent]

        if Y in eCopy.keys():
            return bn.p(Y, eCopy[Y], condition) * enumerate_all(vars, eCopy, bn, sum)
        else:
            for value in bn.domain(Y):
                varsCopy = deepcopy(vars)
                eCopy[Y] = value
                sum += bn.p(Y, eCopy[Y], condition) * enumerate_all(varsCopy, eCopy, bn, sum)
            return sum

    Q = {}
    for xi in bn.domain(X):
        e[X] = xi
        Q[xi] = enumerate_all(topological_sort(bn), e, bn)

    factor = 1/sum(Q.values())
    normalizedQ = {k: v*factor for k, v in Q.items()}
    print(normalizedQ)
    
    return normalizedQ

prob = enumeration_ask("B", {"J": True, "M": True}, bn)
assert abs(prob[True] - 0.284) <= 0.001
assert abs(prob[False] - 0.716) <= 0.001

def coins(pa: float, pb: float, pc: float) -> BayesianNet:
    coins = BayesianNet()
    coins.addUnconditionalVariable('Coins', {'A': 1/3, 'B' : 1/3, 'C' : 1/3})
    coins.addConditionalVariable('X1', ('Coins', ), {('A',) : {'H' : pa, 'T' : 1-pa}, ('B', ) : {'H' : pb, 'T' : 1-pb}, ('C', ) : {'H' : pc, 'T' : 1-pc}})
    coins.addConditionalVariable('X2', ('Coins', ), {('A',) : {'H' : pa, 'T' : 1-pa}, ('B', ) : {'H' : pb, 'T' : 1-pb}, ('C', ) : {'H' : pc, 'T' : 1-pc}})
    coins.addConditionalVariable('X3', ('Coins', ), {('A',) : {'H' : pa, 'T' : 1-pa}, ('B', ) : {'H' : pb, 'T' : 1-pb}, ('C', ) : {'H' : pc, 'T' : 1-pc}})
    return coins

coins_bn = coins(.3, .6, .75)
pd = enumeration_ask('Coins', {'X1': 'H', 'X2': 'H', 'X3': 'T'}, coins_bn)
pd

assert abs(pd['A'] - 0.181) <= 0.005
assert abs(pd['B'] - 0.414) <= 0.005
assert abs(pd['C'] - 0.405) <= 0.005
assert pd['B'] > pd['C'] > pd['A']

def generateThrows(number: int) -> dict:
    random.seed()
    output = {}
    for i in range(number):
        name = 'Throw' + str(i + 1)
        rand = random.randint(0, 1)
        if rand == 0:
            output[name] = 'T'
        else:
            output[name] = 'H'
    return output

def generateCoins(number: int) -> list[float]:
    random.seed()
    output = []
    for _ in range(number):
        output.append(random.uniform(0, 1))
    return output

def iterAllStrings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_uppercase, repeat=size):
            yield "".join(s)

def multipleCoins(probabilities: list[float], throws: dict) -> BayesianNet:
    net = BayesianNet()
    iterator = 0
    coins = {}
    cpd = {}
    for s in itertools.islice(iterAllStrings(), len(probabilities)):
        name = s
        cpd[name] = 1/len(probabilities)
        coins[name] = probabilities[iterator]
        iterator += 1
    cpd['A'] += 1 - sum(cpd.values())   #to help the cpd normalization, but i didn't want to modify your code, and it's such a small precision error
    #that it shouldn't influence the output by a lot in any case.
    net.addUnconditionalVariable('Coins', cpd)

    for i in range(len(throws)):
        name = 'Throw' + str(i+1)
        parentDependencies = {}
        for coin, prob in coins.items():
            parentDependencies[(coin, )] = {'H' : prob, 'T' : 1 - prob}
        net.addConditionalVariable(name, ('Coins', ), parentDependencies)
    return net

coins = [0.3, 0.6, 0.75, 0.5, 0.69, 0.420]
throws = {'Throw1': 'T', 'Throw2': 'T', 'Throw3': 'H', 'Throw4' : 'H', 'Throw5' : 'T'}
coins_bn = multipleCoins(coins, throws)
pd = enumeration_ask('Coins', throws, coins_bn)

# RANDOMLY GENERATED THINGS ARE FUN BUT REGARDLESS OF THAT I THINK THAT MANUAL INPUT IS MORE INSIGHTFUL AND INTERESTING ANYWAYS
# coins = generateCoins(random.randint(2, 10))
# throws = generateThrows(random.randint(2, 10))
# coins_bn = multipleCoins(coins, throws)
# pd = enumeration_ask('Coins', throws, coins_bn)


plt.pie(sorted(pd.values()), labels = [f'Coin {str(i)} ({p:.2f} H)' for i, p in sorted(zip(sorted(pd.keys()), coins), key = lambda item: pd[item[0]])], autopct = '%1.2f%%')
plt.show()
