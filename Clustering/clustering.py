import common as cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython import display

# Computes a Euclidean distance between points A and B
def getEuclideanDistance(A, B):
    A = np.array(A)
    B = np.array(B)
    return(np.linalg.norm(A - B))

data = cm.getTestDataSet()
cm.displayDataSet(plt, data) #plt = plot package; see the imports above
assignments = cm.getTestAssignments() ### GET "TRUE" GROUP ASSIGNMENT
cm.displayDataSet(plt, data, assignments = assignments)

def getCentroids(K, data):
    Centroids = []
    temp1 = data
    np.random.shuffle(temp1)
    temp2 = temp1
    for i in range(K):
        Centroids.append(temp2[i])
    return Centroids

def doKMeansStep(DATA, M, K, CENTROIDS, ASSIGNMENTS):    
    NO_CHANGE = True
    if ASSIGNMENTS is None: NO_CHANGE = False
    NEW_ASSIGNMENTS = [[] for k in range(K)]
    for i in range(len(DATA)):
        dist = float('inf')
        for j in range(K):
            z = getEuclideanDistance(DATA[i],CENTROIDS[j])
            if z < dist:
                dist = z
                temp1 = j
        NEW_ASSIGNMENTS[temp1].append(i)

    NEW_CENTROIDS = [[] for i in range(K)]
    for cluster in range(K):
        for atr in range(M):
            temp2 = 0
            for point in range(len(NEW_ASSIGNMENTS[cluster])):
                temp2 += DATA[NEW_ASSIGNMENTS[cluster][point]][atr]
            avrg = temp2/(len(NEW_ASSIGNMENTS[cluster]))
            NEW_CENTROIDS[cluster].append(avrg)
            
    if NEW_ASSIGNMENTS != ASSIGNMENTS:
        NO_CHANGE = False
    
    return NO_CHANGE, NEW_CENTROIDS, NEW_ASSIGNMENTS

DATA = cm.getTestDataSet()
CENTROIDS = getCentroids(2, DATA)
NO_CHANGE, CENTROIDS, ASSIGNMENTS = doKMeansStep(DATA, 2, 2, CENTROIDS.copy(), None)
cm.displayDataSet(plt, DATA, assignments = ASSIGNMENTS, centroids = CENTROIDS)

def doKMeans(DATA, CENTROIDS, K, M, display = True):
    ASSIGNMENTS = [[] for i in range(K)]
    NO_CHANGE, CENTROIDS, ASSIGNMENTS = doKMeansStep(DATA, M, K, CENTROIDS, ASSIGNMENTS)
    z = 1
    for i in range(99):
        if NO_CHANGE == False:
            z += 1
            NO_CHANGE, CENTROIDS, ASSIGNMENTS = doKMeansStep(DATA, M, K, CENTROIDS, ASSIGNMENTS)
        if NO_CHANGE == True: 
            break
    cm.displayDataSet(plt, DATA, assignments = ASSIGNMENTS, centroids = CENTROIDS)
    print(z)
    return DATA, CENTROIDS, ASSIGNMENTS   

K = 2
M = len(DATA[0])
DATA = cm.getTestDataSet()
CENTROIDS = getCentroids(2, DATA)
DATA, CENTROIDS, ASSIGNMENTS = doKMeans(DATA, CENTROIDS, K, M)

DATA = cm.getTestDataSet()
M = len(DATA[0])


for K in range(2, 11):
    CENTROIDS = getCentroids(K, DATA)
    DATA, NEW_CENTROIDS, NEW_ASSIGNMENTS = doKMeans(DATA, CENTROIDS, K, M)

K = 3
DATA = cm.getTestDataSet()
M = len(DATA[0])
CENTROIDS = getCentroids(K, DATA)


DATA, CENTROIDS, ASSIGNMENTS = doKMeans(DATA, CENTROIDS, K, M)

def getTotalDistance(DATA, CENTROIDS, ASSIGNMENTS):
    temp1 = 0
    for i in range(len(ASSIGNMENTS)):
        for j in range(len(ASSIGNMENTS[i])):
            temp1 += getEuclideanDistance(CENTROIDS[i],DATA[ASSIGNMENTS[i][j]])
    return temp1
getTotalDistance(DATA,CENTROIDS,ASSIGNMENTS)

DATA = cm.getTestDataSet()
M = len(DATA[0])
results = []

def getDirection(point1, point2):
    dire = (point1[1] - point2[1])/(point1[0] - point2[0])
    return dire
def getAngle(dire1, dire2):
    z =np.arctan((abs((dire1 - dire2)/(1+(dire1*dire2)))))
    return z
    
for K in range(2,11):
    DATA
    CENTROIDS = getCentroids(K, DATA)
    DATA, CENTROIDS, ASSIGNMENTS = doKMeans(DATA, CENTROIDS, K, M)
    print(getTotalDistance(DATA, CENTROIDS, ASSIGNMENTS))
    results.append([K,getTotalDistance(DATA, CENTROIDS, ASSIGNMENTS)])
print(results)
cm.displayResults(plt, results)

#Trying to automate finding elbow however it works poorly
directions = []
angles = []
for k in range(len(results)-1):
   directions.append(getDirection(results[k], results[k+1]))
for j in range(len(results)-2):
   angles.append(getAngle(directions[j], directions[j+1]))
print(3 + max([i if angles[i] == max(angles) else -1 for i in range(len(angles))]))

DATA = cm.getCaseDataSet()

M = len(DATA[0])

def getAtributeSet(DATA,Atr):
    AtrSet = []
    for i in range(len(DATA)):
        AtrSet.append(DATA[i][Atr])
    return AtrSet

def Normalize(DATA, Atr):
    global M
    NormalizedAtrSet = []
    for i in range(len(DATA[0])):
        z = (DATA[Atr][i] - min(DATA[Atr]))/(max(DATA[Atr])-min(DATA[Atr]))
        NormalizedAtrSet.append(z)
    return NormalizedAtrSet
    
DATA_N = []
NormalizedGroupedAtr = []
GroupedAtr = []
for m in range(M):
    AtrSet = getAtributeSet(DATA, m)
    GroupedAtr.append(AtrSet)
for m in range(M):
    NormalizedAtrSet = Normalize(GroupedAtr, m)
    NormalizedGroupedAtr.append(NormalizedAtrSet)
    
for i in range(len(DATA)):
    temp1 = []
    for j in range(M):
        z = NormalizedGroupedAtr[j][i]
        temp1.append(z)
    DATA_N.append(temp1)

results_test = []
K = 2
M = len(DATA_N[0])

def doKMeans_CaseStudy(DATA, M, K):
    ASSIGNMENTS = [[] for i in range(K)]
    CENTROIDS = getCentroids(K, DATA)
    NO_CHANGE, CENTROIDS, ASSIGNMENTS = doKMeansStep(DATA, M, K, CENTROIDS, ASSIGNMENTS)
    z = 1
    for i in range(99):
        if NO_CHANGE == False:
            z += 1
            NO_CHANGE, CENTROIDS, ASSIGNMENTS = doKMeansStep(DATA, M, K, CENTROIDS, ASSIGNMENTS)
        if NO_CHANGE == True:
            break
    cm.displayDataSet(plt, DATA, assignments = ASSIGNMENTS, centroids = CENTROIDS)
    print(z)
    print(getTotalDistance(DATA, CENTROIDS, ASSIGNMENTS))
    return DATA, CENTROIDS, ASSIGNMENTS 


DATA_N, CENTROIDS, ASSIGNMENTS = doKMeans_CaseStudy(DATA_N, M, K)

global DATA_N
M = len(DATA_N[0])
result_test = []

for K in range(2, 11):
    DATA_N, CENTROIDS, ASSIGNMENTS = doKMeans_CaseStudy(DATA_N, M, K)
    result_test.append([K,getTotalDistance(DATA_N, CENTROIDS, ASSIGNMENTS)])
print(result_test)
cm.displayResults(plt, result_test)

DATA = cm.getCaseDataSet()


M = len(DATA_N[0])
K = 5

DATA_N, CENTROIDS, ASSIGNMENTS = doKMeans_CaseStudy(DATA_N, M, K)

def mean(assignment, cluster, atr):
    temp = 0
    for j in assignment[cluster]:
        temp += DATA[j][atr]
    sol = temp/len(assignment[cluster])
    return sol
        
def minimal(assignment, cluster, atr):
    temp = []
    for j in assignment[cluster]:
        temp.append(DATA[j][atr])
    sol = min(temp)
    return sol
def maximal(assignment, cluster, atr):
    temp = []
    for j in assignment[cluster]:
        temp.append(DATA[j][atr])
    sol = np.max(temp)
    return sol
def deviation(assignment, cluster, atr):
    temp = []
    for j in assignment[cluster]:
        temp.append(DATA[j][atr])
    sol = np.std(temp)
    return sol

for i in range(M):
    for j in range(K):
        print('Mean'+' cluster:'+str(j+1)+'  attribute:'+str(i+1)+' : '+str(mean(ASSIGNMENTS, j, i)))
        print('Min'+' cluster:'+str(j+1)+'  attribute:'+str(i+1)+' : '+str(minimal(ASSIGNMENTS, j, i)))
        print('Max'+' cluster:'+str(j+1)+'  attribute:'+str(i+1)+' : '+str(maximal(ASSIGNMENTS, j, i)))
        print('Standard Deviation'+' cluster:'+str(j+1)+'  attribute:'+str(i+1)+' : '+str(deviation(ASSIGNMENTS, j, i)))