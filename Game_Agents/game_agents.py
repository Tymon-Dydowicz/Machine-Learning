from IPython import display
from time import sleep
import math

class Game:
    @property
    def initial_state(self):
        ...
        return state
    
    def player(self, state):
        ...
        return playerno
        
    def actions(self, state):
        ...
        return actions
        
    def result(self, state, action):
        ...
        return new_state
        
    def is_terminal(self, state):
        ...
        return boolean
        
    def utility(self, state, player):
        ...        
        return number
        
    def print_state(self, state):
        ...

def opponent(player):    
    assert player in {1, 2}
    if player == 1:
        return 2
    else:
        return 1

class TicTacToe(Game):    
    @property
    def initial_state(self):
        return (1, (0,)*9)
    
    def player(self, state):
        return state[0]
        
    def actions(self, state):
        return [i for i, v in enumerate(state[1]) if v == 0]
        
    def result(self, state, action):
        board = state[1]
        assert board[action] == 0
        assert state[0] in {1, 2}
        board = board[:action] + (state[0],) + board[action+1:]
        next_player = opponent(state[0])        
        return (next_player, board)
        
    def _has_line(self, state, player):
        board = state[1]
        for i in [0, 3, 6]:
            if board[i] == board[i+1] == board[i+2] == player:
                return True
        for i in [0, 1, 2]:
            if board[i] == board[i+3] == board[i+6] == player:
                return True
        if board[0] == board[3+1] == board[2*3+2] == player:
            return True
        if board[2] == board[3+1] == board[2*3] == player:
            return True
        return False
        
    def is_terminal(self, state):
        if all([v != 0 for v in state[1]]):
            return True
        return self._has_line(state, 1) or self._has_line(state, 2)
    
    def utility(self, state, player):
        assert player in {1, 2}
        mine = self._has_line(state, player)
        opponents = self._has_line(state, opponent(player))
        if mine and not opponents:
            return 1
        if not mine and opponents:
            return -1
        return 0    
    
    def print_state(self, state):
        print("Player making move", " OX"[state[0]])
        board = ["_OX"[v] for v in state[1]]
        print(*board[0:3])
        print(*board[3:6])
        print(*board[6:9])

game = TicTacToe()
state = game.initial_state
game.print_state(state)

for action in [4,0,6,2,1,7,5,3,8]:
    assert action in game.actions(state)
    assert not game.is_terminal(state)
    state = game.result(state, action)
    game.print_state(state)
    
print("Reached terminal state?", game.is_terminal(state))
print("Utility for the 1st player", game.utility(state, 1))
print("Utility for the 2nd player", game.utility(state, 2))

game = TicTacToe()
state = game.initial_state
game.print_state(state)

for action in [4,0,6,2,1,8,7]:
    assert action in game.actions(state)
    assert not game.is_terminal(state)
    state = game.result(state, action)
    game.print_state(state)
    
print("Reached terminal state?", game.is_terminal(state))
print("Utility for the 1st player", game.utility(state, 1))
print("Utility for the 2nd player", game.utility(state, 2))

game = TicTacToe()
state = game.initial_state
game.print_state(state)

for action in [2,4,6,0,7,8]:
    assert action in game.actions(state)
    assert not game.is_terminal(state)
    state = game.result(state, action)
    game.print_state(state)
    
print("Reached terminal state?", game.is_terminal(state))
print("Utility for the 1st player", game.utility(state, 1))
print("Utility for the 2nd player", game.utility(state, 2))

def dummy(game, state):
    return game.actions(state)[0]

def judge(game: Game, player1, player2):    
    state = game.initial_state

    while not game.is_terminal(state):
        if game.player(state) == 1:
            action = player1(game, state)
        else:
            action = player2(game, state) 
        display.clear_output(wait = True);        
        game.print_state(state)
        sleep(0.2)
        print("Action:", action)
        print()
        state = game.result(state, action)
     
    game.print_state(state)
    print("Reached terminal state?", game.is_terminal(state))
    u1 = game.utility(state, 1)
    u2 = game.utility(state, 2)
    print("Utility for the 1st player", u1)
    print("Utility for the 2nd player", u2)
    if u1 > u2:
        print("Winner: 1st player")
    elif u1 < u2:
        print("Winner: 2nd player")
    else:
        print("Draw")

judge(TicTacToe(), dummy, dummy)

def maxValue(game: Game, state, player):
    u = 'toMax'
    if game.is_terminal(state):
        return game.utility(state, player), None
    
    for action in game.actions(state):
        u2, move1 = minValue(game, game.result(state, action), player)
        if u == 'toMax' or u2 > u:
            u, move = u2, action
    return u, move

def minValue(game: Game, state, player):
    u = 'toMin'
    if game.is_terminal(state):
        return game.utility(state, player), None
    
    for action in game.actions(state):
        u2, move1 = maxValue(game, game.result(state, action), player)
        if u == 'toMin' or u2 < u:
            u, move = u2, action
    return u, move

def minimax(game: Game, state):
    player = game.player(state)
    value, move = maxValue(game, state, player)
    return move

def maxValueAB(game: Game, state, player, alfa, beta):
    u = 'toMax'
    if game.is_terminal(state):
        return game.utility(state, player), None
    
    for action in game.actions(state):
        u2, move1 = minValueAB(game, game.result(state, action), player, alfa, beta)
        if u == 'toMax' or u2 > u:
            u, move = u2, action
            alfa = max(alfa, u)
        if u >= beta:
            return u, move
    return u, move

def minValueAB(game: Game, state, player, alfa, beta):
    u = 'toMin'
    if game.is_terminal(state):
        return game.utility(state, player), None
    
    for action in game.actions(state):
        u2, move1 = maxValueAB(game, game.result(state, action), player, alfa, beta)
        if u == 'toMin' or u2 < u:
            u, move = u2, action
            beta = min(beta, u)
        if u <= alfa:
            return u, move
    return u, move
def alphabeta(game, state):
    player = game.player(state)
    value, move = maxValueAB(game, state, player, float('-inf'), float('inf'))
    return move

class Migration:
    def __init__(self, n):
        self.n = n
    
    @property
    def initial_state(self):
        board = [[0]*self.n for _ in range(self.n)]
        k = math.ceil(self.n/2 - 1)
        for y in range(k):
            for x in range(y + 1, self.n - y - 1):
                board[x][y] = 1    
        for x in range(k):
            for y in range(x + 1, self.n - x - 1):
                board[self.n - x - 1][y] = 2
        board = tuple((tuple(row) for row in board))
        return (1, board)
    
    def player(self, state):
        return state[0]
    
    def _is_valid(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n
    
    def actions(self, state):
        board = state[1]
        player = self.player(state)
        opp = opponent(player)
        if player == 1:
            dx, dy = 0, 1
        else:
            assert player == 2
            dx, dy = -1, 0
        actions = []
        for x in range(self.n):
            nx = x + dx
            for y in range(self.n):
                ny = y + dy
                if board[x][y] == player and self._is_valid(nx, ny) and board[nx][ny] == 0:
                    actions.append((x, y, nx, ny))
        return actions
    
    def result(self, state, action):
        x, y, nx, ny = action
        player, board = state
        board = [list(row) for row in board]
        assert board[x][y] == player
        assert board[nx][ny] == 0
        board[x][y] = 0
        board[nx][ny] = player
        board = tuple((tuple(row) for row in board))
        return (opponent(player), board)
    
    def is_terminal(self, state):
        return len(self.actions(state)) == 0
        
    def utility(self, state, player):
        assert self.is_terminal(state)
        if self.player(state) == player:
            return -1
        else:
            return 1
        
    def print_state(self, state):
        print("Player making move", "_\u25CB\u25CF"[state[0]])
        for row in state[1]:
            print(*["_\u25CB\u25CF"[v] for v in row])

game = Migration(8)
state = game.initial_state
game.print_state(state)

print(game.actions(state))
move = game.actions(state)[0]
state = game.result(state, move)
game.print_state(state)

move = game.actions(state)[5]
state = game.result(state, move)
game.print_state(state)
judge(Migration(4), alphabeta, alphabeta)

class MigrationWithHeuristic(Migration):
    def evaluate(self, state, player):
        moves = 0
        moves += len(self.actions(state))
        state = list(state)
        state[0] = opponent(player)
        state = tuple(state)
        moves -= len(self.actions(state))
        return moves

class HeuristicAlphaBeta:

    
    def __init__(self, max_depth):
        self.max_depth = max_depth
        
    def __call__(self, game, state):
        return self.alphabetaH(game, state)

    def maxValueABH(self, game: Game, state, player, alfa, beta, depth):
        u = 'toMax'
        depth += 1

        if game.is_terminal(state):
            return game.utility(state, player), None

        if depth > self.max_depth:
            return game.evaluate(state, player), None

        for action in game.actions(state):
            u2, move1 = self.minValueABH(game, game.result(state, action), player, alfa, beta, depth)
            if u == 'toMax' or u2 > u:
                u, move = u2, action
                alfa = max(alfa, u)
            if u >= beta:
                return u, move
        return u, move

    def minValueABH(self, game: Game, state, player, alfa, beta, depth):
        u = 'toMin'
        depth += 1

        if game.is_terminal(state):
            return game.utility(state, player), None
        
        if depth > self.max_depth:
            return game.evaluate(state, player), None
        
        for action in game.actions(state):
            u2, move1 = self.maxValueABH(game, game.result(state, action), player, alfa, beta, depth)
            if u == 'toMin' or u2 < u:
                u, move = u2, action
                beta = min(beta, u)
            if u <= alfa:
                return u, move
        return u, move
    
    def alphabetaH(self, game, state):
        player = game.player(state)
        value, move = self.maxValueABH(game, state, player, float('-inf'), float('inf'), depth = 0)
        return move