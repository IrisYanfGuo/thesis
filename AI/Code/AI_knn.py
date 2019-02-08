#!/usr/local/bin/python3
import numpy as np
from operator import itemgetter
import toolkit as tk 

class knn():

    # read trainSet, testSet and "attributes" 
    def __init__(self,attributes,trainSet):
        self.__attributes = attributes
        self.__trainSet = trainSet
        self.__training()

    # to calculate the Euclidean distance
    def __getEuclideanDistance(self, data1, data2):
        d = 0.0
        for x in range(len(data1)-1):
            d += pow((data1[x] - data2[x]),2)
        return np.sqrt(d)

    # get K near neighborhoods
    #  use list, because dictionay will cause value missing
    def __getKNearNeighbors(self, trainSet, testInstance, n=3):
        distances = []
        neighbors = []
        #neighbors2 = []
        #dis = {}

        for i in range(len(trainSet)):
            dist = self.__getEuclideanDistance(testInstance,trainSet[i])
            distances.append((trainSet[i],dist))

        distances.sort(key=itemgetter(1))

        for i in range(n):
            neighbors.append(distances[i][0])
        return neighbors


    def __getClassification(self, neighbors):
        dic = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]   # result
            dic.setdefault(response,dic.get(response,0)+1)
        temp = sorted(dic.items(),key=itemgetter(1))
        temp.reverse()
        #choose the best one
        return temp[0][0]


    def __training(self):
        # this is for double check, because we need numeric
        for i in range(len(self.__trainSet)):
            for j in range(len(self.__attributes)):
                self.__trainSet[i][j] = float(self.__trainSet[i][j])

    def getPrediction(self,testSet,k=3):
        for i in range(len(testSet)):
            for j in range(len(self.__attributes)):
                testSet[i][j] = float(testSet[i][j])

        predictions = []
        
        for i in range(len(testSet)):
            neighbors = self.__getKNearNeighbors(self.__trainSet,testSet[i],k)
            result = self.__getClassification(neighbors)
            predictions.append(result)


        if (len(testSet[0]) == len(self.__trainSet)):
            Accuracy = np.nan
        else:
            correct = 0
            for i in range(len(testSet)):
                if testSet[i][-1] == predictions[i]:
                    correct += 1
            Accuracy = correct/float(len(testSet)) *100.0

        return Accuracy,predictions



