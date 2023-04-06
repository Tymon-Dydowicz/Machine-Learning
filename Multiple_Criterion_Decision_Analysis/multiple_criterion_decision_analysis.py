import pandas as pd
import numpy as np
import common as cm
import matplotlib.pyplot as plt

data = {
    'mode': ['bus', 'bike', 'car', 'train', 'foot'],
    'time': [6, 8, 2, 3, 10],
    'comfort': [3, 2, 10, 5, 2],
    'price': [6, 2, 9, 5, 0],
    'reliability': [2, 8, 7, 6, 5]
}

criteria = ['time', 'comfort', 'price', 'reliability']

data = pd.DataFrame(data, columns=['mode', 'time', 'comfort', 'price', 'reliability'])
parameters = {'time': {'weight': 4, 'q': 1.0, 'p': 2, 'v': 4, 'type': 'cost'},
 'comfort': {'weight': 2, 'q': 2.0, 'p': 3, 'v': 6, 'type': 'gain'},
 'price': {'weight': 3, 'q': 1.0, 'p': 3, 'v': 5, 'type': 'cost'},
 'reliability': {'weight': 1, 'q': 1.5, 'p': 3, 'v': 5, 'type': 'gain'}}

sum_weights = 10.

pd.DataFrame(parameters, columns=['time', 'comfort', 'price','reliability']).reindex(['type', 'q', 'p', 'v', 'weight']).T

def getConcordanceCost(gA, gB, q, p):
    return getConcordanceGain(gB, gA, q, p)

def getConcordanceGain(gA, gB, q, p):
    if gA - gB >= -q:
      return 1
    elif gA - gB <= -p:
      return 0
    else:
      return (p - (gB - gA))/(p - q)

def getComprehensiveConcordance(A, B, criteria, parameters):
    concordance = 0.0
    sum_weights = 0.0
    for criterion in criteria:
      if parameters[criterion]['type'] == 'gain':
        gain = getConcordanceGain(A[criterion], B[criterion], parameters[criterion]['q'], parameters[criterion]['p']) * parameters[criterion]['weight']
        concordance += gain
        sum_weights += parameters[criterion]['weight']
      elif parameters[criterion]['type'] == 'cost':
        cost = getConcordanceCost(A[criterion], B[criterion], parameters[criterion]['q'], parameters[criterion]['p']) * parameters[criterion]['weight']
        concordance += cost
        sum_weights += parameters[criterion]['weight']
    return concordance / sum_weights

for alternative_id, alternative_row in data.iterrows():
    print("C({0},{1}) = ".format(0, alternative_id), getComprehensiveConcordance(data.loc[0], alternative_row, criteria, parameters))

def getConcordanceMatrix(data, criteria, parameters, majority_treshold=0.7):
    concordance_matrix = np.zeros((len(data),len(data)))
    for A_idx, A_row in data.iterrows():
        for B_idx, B_row in data.iterrows():
            CompConc = getComprehensiveConcordance(A_row, B_row, criteria, parameters)
            if A_row['mode'] == B_row['mode']:
              continue
            elif CompConc >= majority_treshold:
              concordance_matrix[A_idx, B_idx] = 1
            else:
              continue
    return concordance_matrix

print(getConcordanceMatrix(data, criteria, parameters))

def getDiscordanceGain(gA, gB, v):
    if gB - gA >= v:
      return 1
    else:
      return 0

def getDiscordanceCost(gA, gB, v):
    return getDiscordanceGain(gB, gA, v)

def getComprehensiveDiscordance(A, B, criteria, parameters):
    temp = 0
    for criterion in criteria:
      if parameters[criterion]['type'] == 'gain':
        temp += getDiscordanceGain(A[criterion], B[criterion], parameters[criterion]['v'])
      elif parameters[criterion]['type'] == 'cost':
        temp += getDiscordanceCost(A[criterion], B[criterion], parameters[criterion]['v'])
    if temp > 0:
      return 1
    else:
      return 0.

for alternative_id, alternative_row in data.iterrows():
    print("D({0},{1}) = ".format(0, alternative_id),getComprehensiveDiscordance(data.loc[0], alternative_row, criteria, parameters))

def getDiscordanceMatrix(data, criteria, parameters):
    discordance_matrix = np.zeros((len(data),len(data)))
    for A_idx, A_row in data.iterrows():
        for B_idx, B_row in data.iterrows():
            if A_idx != B_idx:
                CompDisc = getComprehensiveDiscordance(A_row, B_row, criteria, parameters)
                if CompDisc == 1:
                  discordance_matrix[A_idx, B_idx] = 1
                else:
                  continue        
    return discordance_matrix

getDiscordanceMatrix(data, criteria, parameters)

def getOutrankingMatrix(data, criteria, parameters, majority_treshold):
    concordance_matrix = getConcordanceMatrix(data, criteria, parameters, majority_treshold)
    discordance_matrix = getDiscordanceMatrix(data, criteria, parameters)
    n = len(data)
    outranking_matrix = np.zeros((n,n))
    for A_idx in range(n):
      for B_idx in range(n):
        if concordance_matrix[A_idx, B_idx] == 1 and discordance_matrix[A_idx, B_idx] == 0:
          outranking_matrix[A_idx, B_idx] = 1
        else: 
          continue
    return outranking_matrix

outranking_matrix = getOutrankingMatrix(data, criteria, parameters, majority_treshold=0.75)

def toAdjacencyList(outranking_matrix):
    n = len(outranking_matrix)
    graph = {i:[] for i in range(n)}
    for j in range(n):
      for k in range(n):
        if outranking_matrix[j][k] == 1:
          graph[j].append(k)
        else:
          continue
    return graph

graph = toAdjacencyList(outranking_matrix)
print(graph)

cm.PrintGraph(graph, filename="graph_part_2")
plt.imshow(plt.imread("graph_part_2.png"))

kernel = []
reverse_graph = getReverseGraph(graph)
Kernel(reverse_graph, kernel)

outranking_matrix = np.array(   [[0., 1., 0., 0., 0., 0., 1., 0.],
                                 [0., 0., 1., 0., 1., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1.],
                                 [0., 0., 0., 0., 1., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1.],
                                 [0., 0., 0., 0., 0., 0., 0., 0.]])

graph = toAdjacencyList(outranking_matrix)

graph


cm.PrintGraph(graph, filename="graph_part_3")
plt.imshow(plt.imread("graph_part_3.png"))
def getReverseGraph(graph):
    n = len(graph)
    reverse_graph = {i:[] for i in graph}
    outrank = np.zeros((n, n))
    for i in graph:
      for j in graph[i]:
        outrank[i, j] = 1
    for j in range(n):
      for k in range(n):
        if outrank[k][j] == 1:
          reverse_graph[j].append(k)
        else:
          continue
    return reverse_graph

reverse_graph = getReverseGraph(graph)
reverse_graph

def getKernel(reverse_graph, kernel):
  for node in reverse_graph:
    if len(reverse_graph[node]) == 0:
      kernel.append(node)
  return kernel
def getNewGraph(reverse_graph, kernel):
  new_reverse_graph = {}
  for node in reverse_graph:
    temp1 = 0
    if node not in kernel:
      for j in reverse_graph[node]:
        if j in kernel:
          temp1 +=1
        else:
          continue
      if temp1 == 0:
        new_reverse_graph[node] = reverse_graph[node]
      else:
        continue
  else:
    pass
  return new_reverse_graph
def clearGraph(graph):
  for node in graph:
    temp2 = 0
    for j in graph[node]:
      if j not in graph.keys():
        graph[node].remove(j)
  return graph

kernel = []
def Kernel(graph, kernel):
    getKernel(graph, kernel)
    new_graph1 = getNewGraph(graph, kernel)
    new_graph = clearGraph(new_graph1)
    if len(new_graph) == 0:
      return kernel
    else:
      return Kernel(new_graph, kernel)

kernel = []
reverse_graph = getReverseGraph(graph)
Kernel(reverse_graph, kernel)