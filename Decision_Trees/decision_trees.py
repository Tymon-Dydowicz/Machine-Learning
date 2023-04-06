import numpy as np
import math
import matplotlib.pyplot as plt 
import pandas as pd
import common as cm
import statistics
from statistics import mode
from collections import Counter
import sys, os
from re import template
from ssl import DefaultVerifyPaths
from PIL.Image import NONE

attributeNames = ["attr 1", "attr 2", "attr 3", "attr 4", "attr 5"]

data = pd.DataFrame(
    [
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1],
    ],
    columns=attributeNames,
)
data["cl"] = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]

def getEntropy(cl):
    All = len(cl)
    P1, P0 = 0, 0
    for value in cl:
        if value == 1:
            P1 += 1
        elif value == 0:
            P0 += 1
    if P1 == 0:
        Ent = -((P0/All)*np.log2(P0/All))
    elif P0 == 0:
        Ent = -((P1/All)*np.log2(P1/All))
    else:
        Ent = -((P1/All)*np.log2(P1/All)) - ((P0/All)*np.log2(P0/All))
    return Ent

def getConditionalEntropy(cl, attr):
    P0 = cl[attr == 0]
    P1 = cl[attr == 1]
    All = len(cl)
    temp1 = len(cl[attr == 1])
    temp0 = len(cl[attr == 0])
    if len(P0) == 0:
        Cond = (temp1/All)*getEntropy(P1)
    elif len(P1) == 0:
        Cond = (temp0/All)*getEntropy(P0)
    else:
        Cond = (temp1/All)*getEntropy(P1) + (temp0/All)*getEntropy(P0)
    return Cond

def getInformationGain(cl, attr):
    Gain = getEntropy(cl) - getConditionalEntropy(cl, attr)
    return Gain

class Node:
    def __init__(self, attr, left, right, value):
        self.attr = attr
        self.left = left
        self.right = right
        self.value = value

    def __call__(self, obj):
        if self.value is None:
            if obj[self.attr] == 0:
                return self.left(obj)
            else:
                return self.right(obj)
        else:
            return self.value
        
### EXAMPLE
def example(obj):
    root = Node(0, None, None, None) ###  IN ROOT SPLIT ON 1ST (0) ATTRIBUTE
    lChildren = Node(1, None, None, None) ### IN ROOT's LEFT CHILDREN SPLIT ON 2ND (1) ATTRIBUTE
    rChildren = Node(None, None, None, 2) ### IN ROOT's RIGHT CHILDREN -> DECISION = 2
    root.left = lChildren
    root.right = rChildren
    llChildren = Node(None, None, None, 3) ### IN ROOT's LEFT-LEFT CHILDREN -> DECISION = 3
    lrChildren = Node(None, None, None, 4) ### IN ROOT's LEFT-RIGHT CHILDREN -> DECISION = 4
    lChildren.left = llChildren
    lChildren.right = lrChildren
    print(root(obj))

root = Node(None, None, None, 1)

cm.getErrorRate(root, data)

cm.printGraph(root, data, fileName = 'DecTree0')
plt.imshow(plt.imread('DecTree0.png'))

def printInformationGain(data , names):
    for attribute_name in names:
        temp1 = getInformationGain(data['cl'], data[attribute_name])
        print(str(temp1) + ' ' + str(attribute_name))
def getBestInformationGain(data, names):
    best = 0
    mem = None
    for attribute_name in names:
        temp1 = (getInformationGain(data['cl'], data[attribute_name]))
        if temp1 > best:
          best = temp1
          mem = attribute_name
        else:
          continue
    if mem is None:
      return "No split will provide new information"       

    #elif best < 'prepruning factor':         # PREPRUNING /OPTIONAL/
   #  return "No split should be performed"
    else:
      return mem
def Most_common(Class):
      temp1 = Counter(Class)
      return temp1.most_common(1)[0][0]

chosen0 = getBestInformationGain(data, attributeNames)
root.attr = chosen0
root.value = None
L1Child = Node(None, None, None, Most_common(data[data[chosen0]==0]['cl']))
R1Child = Node(None, None, None, Most_common(data[data[chosen0]==1]['cl']))
root.left = L1Child
root.right = R1Child

cm.printGraph(root, data, fileName = 'DecTree1')
plt.imshow(plt.imread('DecTree1.png'))
cm.getErrorRate(root, data)

left_data = data[data[chosen0]==0]
right_data = data[data[chosen0]==1]
print(left_data)
print('\n')
print(right_data)

print(left_data)
printInformationGain(left_data, attributeNames)
getBestInformationGain(left_data, attributeNames)

chosen1 = getBestInformationGain(left_data, attributeNames)
L1Child.attr = chosen1
L1Child.value = None
L2Child = Node(None, None, None, Most_common(left_data[left_data[chosen1]==0]['cl']))
L1R1Child = Node(None, None, None, Most_common(left_data[left_data[chosen1]==1]['cl']))
L1Child.left = L2Child
L1Child.right = L1R1Child

cm.printGraph(root, data, fileName = 'DecTree2')
plt.imshow(plt.imread('DecTree2.png'))
cm.getErrorRate(root, data)

left_left_data = left_data[left_data[chosen1]==0]
right_left_data = left_data[left_data[chosen1]==1]
print(left_left_data)
print(right_left_data)

print(right_data)
printInformationGain(right_data, attributeNames)
getBestInformationGain(right_data, attributeNames)

chosen2 = getBestInformationGain(right_data, attributeNames)
R1Child.attr = chosen2
R1Child.value = None
R2Child = Node(None, None, None, Most_common(right_data[right_data[chosen2]==1]['cl']))
R1L1Child = Node(None, None, None, Most_common(right_data[right_data[chosen2]==0]['cl']))
R1Child.right = R2Child
R1Child.left = R1L1Child

cm.printGraph(root, data, fileName = 'DecTree3')
plt.imshow(plt.imread('DecTree3.png'))
cm.getErrorRate(root, data)

left_right_data = right_data[right_data[chosen2]==0]
right_right_data = right_data[right_data[chosen2]==1]
print(left_right_data)
print(right_right_data)
printInformationGain(left_left_data, attributeNames)
getBestInformationGain(left_left_data, attributeNames)

printInformationGain(right_left_data, attributeNames)
getBestInformationGain(right_left_data, attributeNames)

chosen3 = getBestInformationGain(right_left_data, attributeNames)
L1R1Child.attr = chosen3
L1R1Child.value = None
L1R2Child = Node(None, None, None, Most_common(right_left_data[right_left_data[chosen3]==1]['cl']))
L1R1L1Child = Node(None, None, None, Most_common(right_left_data[right_left_data[chosen3]==0]['cl']))
L1R1Child.right = L1R2Child
L1R1Child.left = L1R1L1Child

cm.printGraph(root, data, fileName = 'DecTree4')
plt.imshow(plt.imread('DecTree4.png'))
cm.getErrorRate(root, data)

printInformationGain(left_right_data, attributeNames)
printInformationGain(right_right_data, attributeNames)

depth = 0
def Naming(node, branch):
  node = str(node)
  new_node = node[:-4] + str(branch) + node[-4:]
  return new_node
  
name = 'root'
root = Node(None, None, None, Most_common(data['cl']))

def createTree(data, attributeNames, root, name, depth):
  if depth < max_depth:
    data = data.reset_index().drop("index", axis=1)
    chosen = getBestInformationGain(data, attributeNames)
    if chosen == 'No split will provide new information' or chosen == 'No split should be performed':
        #print("Split wasn't performed at " + str(name))  #Fun to see where the split doesnt happen,
                                                          #Although takes a lot of space in last exercise :C
        pass
    else:
        L_data = data[data[chosen]==0]
        R_data = data[data[chosen]==1]
        root.attr = chosen
        root.value = None
        tempL = Naming(name, 'L')
        tempR = Naming(name, 'R')
        Lnode = tempL
        Rnode = tempR
        Lnode = Node(None, None, None, Most_common(L_data['cl']))
        Rnode = Node(None, None, None, Most_common(R_data['cl']))
        root.left = Lnode
        root.right = Rnode

        depth += 1
        createTree(L_data, attributeNames, Lnode, tempL, depth)
        createTree(R_data, attributeNames, Rnode, tempR, depth)
  else:
    return 'Maximum depth exceeded'

# Training dataset
max_depth = 10
depth = 0
train_attributeNames, train_data = cm.getTrainingDataSet()
root = Node(None, None, None, Most_common(train_data['cl']))
name = 'root'
TreeTest = createTree(train_data, train_attributeNames, root, name, depth)
cm.printGraph(root, train_data, fileName = 'TrainDecTree')
plt.imshow(plt.imread('TrainDecTree.png'))

# Validation dataset
max_depth = 10
valid_attributeNames, valid_data = cm.getValidationDataSet()
print(cm.getErrorRate(root, train_data))
print(cm.getErrorRate(root, valid_data))

for i in range(10):
    max_depth = i
    depth = 0
    root = Node(None, None, None, Most_common(train_data['cl']))
    name = 'root'
    TreeValid = createTree(train_data, train_attributeNames, root, name, depth)
    print(cm.getErrorRate(root, valid_data), '| depth ' + str(i+1))

#for error in Errors:
  #print(error)