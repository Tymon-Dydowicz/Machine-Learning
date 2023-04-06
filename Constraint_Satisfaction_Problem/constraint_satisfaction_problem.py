from queue import PriorityQueue
import copy
from numpy import random
random.seed(151936)

class CSP:
    def __init__(self):
        self.domains = {}
        self.binary = []
    
    def addVariable(self, var, domain):        
        assert var not in self.domains
        self.domains[var] = set(domain)
        
    def addBinaryConstraint(self, var1, operator, var2):
        assert var1 in self.domains
        assert var2 in self.domains
        c = (var1, operator, var2)
        self.binary.append(c)      
        
    def verify(self, left, op, right):
        if op[0] == '=':
            return left == right
        if op == '!=':
            return left != right
        if op == '<':
            return left < right
        if op == '<=':
            return left <= right
        if op == '>':
            return left > right
        if op == '>=':
            return left >= right
        
    def is_complete(self, assignment):
        return self.domains.keys() <= assignment.keys() 
        
    def is_consistent(self, assignment):
        for var, value in assignment.items():            
            if value not in self.domains[var]:
                return False
        for var1, op, var2 in self.binary:
            if var1 in assignment and var2 in assignment:
                if not self.verify(assignment[var1], op, assignment[var2]):
                    return False
        return True

australia = CSP()
australia.addVariable('WA', {'R', 'G', 'B'})
australia.addVariable('NT', {'R', 'G', 'B'})
australia.addVariable('SA', {'R', 'G', 'B'})
australia.addVariable('Q', {'R', 'G', 'B'})
australia.addVariable('NSW', {'R', 'G', 'B'})
australia.addVariable('V', {'R', 'G', 'B'})
australia.addVariable('T', {'R', 'G', 'B'})

australia.addBinaryConstraint('WA', '!=', 'NT')
australia.addBinaryConstraint('WA', '!=', 'SA')
australia.addBinaryConstraint('NT', '!=', 'SA')
australia.addBinaryConstraint('NT', '!=', 'Q')
australia.addBinaryConstraint('SA', '!=', 'Q')
australia.addBinaryConstraint('SA', '!=', 'NSW')
australia.addBinaryConstraint('SA', '!=', 'V')
australia.addBinaryConstraint('Q', '!=', 'NSW')
australia.addBinaryConstraint('NSW', '!=', 'V')

australia.is_consistent({'WA': 'R'})

australia.is_complete({'WA': 'R'})

class RecursiveBacktracking:
    def __init__(self, csp):
        self.csp = csp
        self.assignment = {}
        self.counter = 0
        
    def select_unassigned_variable(self, assignment):
        for var in self.csp.domains.keys():
            if var not in assignment:
                return var
        return None
    
    def order_domain_values(self, variable):
        return self.csp.domains[variable]
        
    
    def solve(self):
        if self.csp.is_complete(self.assignment):
            return self.assignment

        var = self.select_unassigned_variable(self.assignment)
        for value in self.order_domain_values(var):
            self.counter += 1
            self.assignment[var] = value
            if self.csp.is_consistent(self.assignment):
                self.assignment[var] = value
                self.solve()
                if self.csp.is_complete(self.assignment):
                    return self.assignment
            self.assignment.pop(var)
        return False
    
solver = RecursiveBacktracking(australia)
assignment = solver.solve()
print("Assignment", assignment)
print("Is consistent?", australia.is_consistent(assignment))
print("Is complete?", australia.is_complete(assignment))
print("# considered assignments", solver.counter)

class RecursiveBacktrackingWithHeuristics:
    def __init__(self, csp):
        self.csp = csp
        self.assignment = {}
        self.counter = 0

    def select_unassigned_variable(self, assignment, domains):
        memory = (None, 'legalValues', -1)
        for var in self.csp.domains.keys():
            legalValues = domains[var]
            degree = 0

            for constraint in self.csp.binary:
                if var in constraint:
                    if constraint[2 - constraint.index(var)] in self.assignment.keys():
                        if self.assignment[constraint[2 - constraint.index(var)]] in legalValues:
                            legalValues.remove(self.assignment[constraint[2 - constraint.index(var)]])
                    if constraint[2 - constraint.index(var)] not in self.assignment.keys():
                        degree += 1

            if var not in assignment and (memory[1] == 'legalValues' or len(legalValues) <= memory[1]):
                if memory[1] == 'legalValues':
                    memory = (var, len(legalValues), degree)
                elif len(legalValues) < memory[1]:
                    memory = (var, len(legalValues), degree)
                else:
                    if degree > memory[2]:
                        memory = (var, len(legalValues), degree)

        return memory[0]
    
    def order_domain_values(self, variable, domains):
        return domains[variable]
        
    def solve(self, domains):
        if domains == None:
            domainsCopy = copy.deepcopy(self.csp.domains)
        else:
            domainsCopy = copy.deepcopy(domains)

        if self.csp.is_complete(self.assignment):
            return self.assignment

        var = self.select_unassigned_variable(self.assignment, domainsCopy)
        for value in self.order_domain_values(var, domainsCopy):
            domainsCopy2 = copy.deepcopy(domainsCopy)
            self.counter += 1
            self.assignment[var] = value        
            if self.csp.is_consistent(self.assignment):
                self.assignment[var] = value
                domainsCopy2[var] = {value}
                self.solve(domainsCopy2)
                if self.csp.is_complete(self.assignment):
                    return self.assignment
            self.assignment.pop(var)
        return False
        ...

solver = RecursiveBacktrackingWithHeuristics(australia)
assignment = solver.solve(None)
print("Assignment", assignment)
print("Is consistent?", australia.is_consistent(assignment))
print("Is complete?", australia.is_complete(assignment))
print("# considered assignments", solver.counter)

puzzles = ['''
__3_2_6__
9__3_5__1
__18_64__
__81_29__
7_______8
__67_82__
__26_95__
8__2_3__9
__5_1_3__
'''
,
'''
8________
__36_____
_7__9_2__
_5___7___
____457__
___1___3_
__1____68
__85___1_
_9____4__
'''
,
'''
____1__2_
2__3____8
___8_245_
8_32__7__
_________
__6__53_2
_376_8___
4____1__9
_9__3____
'''
,
'''
4_____6__
_7_3_____
_2_5_1___
_9___682_
3_______9
_489___5_
___7_5_1_
_____4_8_
__7_____2
'''
,
'''
_5_8__3__
_48_2____
_9_6_5___
8___7_5__
7_______9
__9_3___2
___3_1_8_
____6_41_
__3__8_5_
''']

solution1 = '''
483921657
967345821
251876493
548132976
729564138
136798245
372689514
814253769
695417382
'''

def setUpSudoku(puzzle):
    sudoku = CSP()
    for i in range(len(puzzle)):
        if puzzle[i] == '_' :
            sudoku.addVariable((1 + i//10, i%10),{str(i) for i in range(1, 10)})
        elif puzzle[i] != '\n':
            sudoku.addVariable((1 + i//10, i%10),{puzzle[i]})

    for i in range(len(puzzle)):
        row = 1 + i//10
        column = i%10
        for j in range(1, 10):
            if column != j and puzzle[i] != '\n' and ((row, j), '!=', (row, column)) not in sudoku.binary:
                sudoku.addBinaryConstraint((row, column), '!=', (row, j))

            if row != j and puzzle[i] != '\n' and ((j, column), '!=', (row, column)) not in sudoku.binary:
                sudoku.addBinaryConstraint((row, column), '!=', (j, column))
            
        if puzzle[i] != '\n':
            row -= 1
            column -= 1
            for k in range(row-row%3, row-row%3 + 3):
                for l in range(column-column%3, column-column%3 + 3):
                    r = k + 1
                    c = l + 1
                    if r <= 9 and c <= 9 and ((r, c), '!=', (row + 1, column + 1)) not in sudoku.binary and ((row + 1, column + 1), '!=', (r, c)) not in sudoku.binary and (r, c) != (row + 1, column + 1):
                        sudoku.addBinaryConstraint((row + 1, column + 1), '!=', (r, c))
    return sudoku

                    
def drawSudoku(assignment):
    print("\n-------------------------")

    for i in range(1, 10):
        for j in range(1, 10):
            if assignment[(i, j)] is not None:
                if j == 0:
                    print("|", end=" ")
                print(f"{assignment[(i, j)]} ", end="")
            if (j) % 3 == 0:
                print("|", end=" ")
        if (i) % 3 == 0:
            print("\n-------------------------", end=" ")
        print()

def reverseOperator(op):
    if op[0] == '=':
        return '='
    if op == '!=':
        return '!='
    if op == '<':
        return '>'
    if op == '<=':
        return '>='
    if op == '>':
        return '<'
    if op == '>=':
        return '<='

sudoku = setUpSudoku(puzzles[0])
solver = RecursiveBacktrackingWithHeuristics(sudoku)
assignment = solver.solve(None)
print("Assignment", assignment)
drawSudoku(assignment)
print("Is consistent?", sudoku.is_consistent(assignment))
print("Is complete?", sudoku.is_complete(assignment))
print("# considered assignments", solver.counter)

class RecursiveBacktrackingWithAC3:
    def __init__(self, csp):
        self.csp = csp
        self.assignment = {}
        self.counter = 0

    def select_unassigned_variable(self, assignment, domains):
        memory = (None, 'legalValues', -1)
        for var in self.csp.domains.keys():
            legalValues = domains[var]
            degree = 0

            for constraint in self.csp.binary:
                if var in constraint:
                    if constraint[2 - constraint.index(var)] in self.assignment.keys():
                        if self.assignment[constraint[2 - constraint.index(var)]] in legalValues:
                            legalValues.remove(self.assignment[constraint[2 - constraint.index(var)]])

                    if constraint[2 - constraint.index(var)] not in self.assignment.keys():
                        degree += 1

            if var not in assignment and (memory[1] == 'legalValues' or len(legalValues) <= memory[1]):
                if memory[1] == 'legalValues':
                    memory = (var, len(legalValues), degree)
                elif len(legalValues) < memory[1]:
                    memory = (var, len(legalValues), degree)
                else:
                    if degree > memory[2]:
                        memory = (var, len(legalValues), degree)

        return memory[0]
    
    def order_domain_values(self, variable, domains):
        return domains[variable]

    def removeInconsistentValues(self, domain, Xi, op, Xj):
        removed = False
        memory = []
        for x in domain[Xi]:
            temp = 0
            for y in domain[Xj]:
                if self.csp.verify(x, op, y):
                    temp += 1
            if temp == 0:
                removed = True
                memory.append(x)
        for x in memory:
            domain[Xi].remove(x)
        return removed


    def AC3(self, domain):
        queue = [i for i in self.csp.binary]
        for Xi, op, Xj in self.csp.binary:
            queue.append((Xj, reverseOperator(op), Xi))

        while len(queue) != 0:
            Xi, op, Xj = queue.pop(0)
            if self.removeInconsistentValues(domain, Xi, op, Xj):
                for constraint in self.csp.binary:
                    if Xi in constraint:
                        queue.append(constraint)
        
    def solve(self, domains):
        if domains == None:
            domainsCopy = copy.deepcopy(self.csp.domains)
        else:
            domainsCopy = copy.deepcopy(domains)

        self.AC3(domainsCopy)
        

        if self.csp.is_complete(self.assignment):
            return self.assignment

        var = self.select_unassigned_variable(self.assignment, domainsCopy)

        for value in self.order_domain_values(var, domainsCopy):
            domainsCopy2 = copy.deepcopy(domainsCopy)
            self.counter += 1
            self.assignment[var] = value
            if self.csp.is_consistent(self.assignment):
                self.assignment[var] = value
                domainsCopy2[var] = {value}
                self.solve(domainsCopy2)
                if self.csp.is_complete(self.assignment):
                    return self.assignment
            self.assignment.pop(var)
        return False

solver = RecursiveBacktrackingWithAC3(australia)
assignment = solver.solve(None)
print("Assignment", assignment)
print("Is consistent?", australia.is_consistent(assignment))
print("Is complete?", australia.is_complete(assignment))
print("# considered assignments", solver.counter)

sudoku = setUpSudoku(puzzles[0])
solver = RecursiveBacktrackingWithAC3(sudoku)
assignment = solver.solve(None)
print("Assignment", assignment)
drawSudoku(assignment)
print("Is consistent?", sudoku.is_consistent(assignment))
print("Is complete?", sudoku.is_complete(assignment))
print("# considered assignments", solver.counter)