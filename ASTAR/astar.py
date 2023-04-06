import numpy as np
import matplotlib.pyplot as plt
import math
import common as cm

position_start, position_target, terrain  = cm.getSmallExample()
### THE NEXT 2 LINES: OVERRIDE SOME CELLS IN THE MATRIX TO SHOW START/STOP LOCATIONS
terrain[position_start[0]][position_start[1]] = 2 ### START
terrain[position_target[0]][position_target[1]] = 2 ### TARGET
### CYAN LOCATIONS = OBSTACLES
plt.imshow(terrain) 
print(str(position_start) + " " + str(position_target))

position_start, position_target, terrain  = cm.getBigExample()
terrain[position_start[0]][position_start[1]] = 2 ### START
terrain[position_target[0]][position_target[1]] = 2 ### START
plt.imshow(terrain)

possible_moves = [(i-1,j-1) for i in range(3) for j in range(3) if i !=1 or j !=1]
#possible_moves = [(0,1),(1,0),(0,-1),(-1,0)]
#possible_moves = [(1,2),(2,1),(1,-2),(-2,1),(-1,2),(2,-1),(-1,-2),(-2,-1)]

possible_moves_costs = [math.sqrt(abs(i)+abs(j)) for i,j in possible_moves]
#possible_moves_costs = [0.5+(abs(i)+abs(j))/2 for i,j in possible_moves]

class Node():
    def __init__(self, parent, position):
        self.parent = parent # a reference to the previous (parent) node in the path
        self.position = position # node's (x, y) position in the map       
        self.g = 0 # g is the actual cost from the source to the node,
        self.f = 0 # f is the estimation of the length of the path from the source to the target, 
                   # passing through the node (i.e., g(x) + h(x)).
        self.step = 0 # it says in which step the node was created (visited).
        
    def __eq__(self, other): ### to check whether two nodes are equal, i.e., takes the same positions
        return self.position == other.position 
    
    def updateG(self, cost): ### update the cost based on its parent's cost. 
        self.g = self.parent.g + cost        
        
    def getPath(self): ### get a path (list of positions) from the source to this node
                       ### this is a recusive method calling self.parent.getPath()
        if self.parent is None:
            return [self.position]
        else:
            return self.parent.getPath() + [self.position]
        
    def estimatePathLenght(self, target):      
        #TODO
        act = self.g
        heur = math.sqrt((self.position[0]-target.position[0])**2+(self.position[1]-target.position[1])**2)
        self.f = act + heur

def validPosition(position, terrain):
    if position[0]<0 or position[1]<0 or position[0]>=terrain.shape[0] or position[1]>=terrain.shape[1]:
        return False
    if  terrain[position] == 1:
        return False
    return True

def A_star(terrain, position_start, position_target, possible_moves, possible_moves_costs):
    root = Node(None, position_start) ### THE SOURCE NODE
    target = Node(None,position_target) ### THE TARGET NODE
    
    closed_set = []
    open_set = []
    open_set.append(root)
    ### Assume that open_set (nodes to be potentially visited) is kept sorted 
    ### according to node's f score (the lower the better). In each iteration, 
    ### the best node is popped from the queue. 
    step = 0 ### This is the first iteration
    root.step = step 
    while(len(open_set)>0):
        ### TODO you can follow the hints or you can write your own version
        open_set.sort(key= lambda x: x.f, reverse=True)
        node = open_set.pop()
        closed_set.append(node)
        step+=2
        node.step = step
        if node == target:
            return node,closed_set
        # Iterate over all possible moves and their costs (you can use zip in the for loop)
        for move, cost in zip(possible_moves, possible_moves_costs):
            position = (node.position[0] + move[0], node.position[1] + move[1])
            if validPosition(position, terrain):
              child = Node(node, position)
              child.updateG(cost)
              if child in closed_set:
                continue
              else:
                if child in open_set:
                  if child.g > open_set[open_set.index(child)].g:
                    continue
                  else:
                    child.estimatePathLenght(target)
                    open_set[open_set.index(child)] = child
                else:
                  child.estimatePathLenght(target)
                  open_set.append(child)
            else:
              continue

position_start, position_target, terrain  = cm.getSmallExample()

plt.imshow(terrain)

path, closed_set = A_star(terrain, position_start, position_target, possible_moves, possible_moves_costs)
print(path.g)
print(path.step)
print(closed_set)

cm.plotPath(terrain, path)

cm.plotSteps(terrain, closed_set)

position_start, position_target, terrain  = cm.getBigExample()

path,closed_set = A_star(terrain, position_start, position_target, possible_moves, possible_moves_costs)
print(path.g)
print(path.step)

cm.plotPath(terrain, path)

cm.plotSteps(terrain,closed_set)

