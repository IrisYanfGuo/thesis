import math
import operator
from random import random
import toolkit as tk


# iris, continuous variables
# car, discrete variables
class Naive_bayes(object):


    def __init__(self, attr,trainset):

        self.__attributes = attr
        self.__trainSet = trainset

        self.__pc = {}
        self.__px = []
        self.__pxc = {}

        self.train()

    def getAttributes(self):
        return self.__attributes
        # get instances

    def getInstances(self):
        return self.__instances

    def printTestset(self):
        for i in self.__testSet:
            print(i)

    def printTrainset(self):
        for i in self.__trainSet:
            print(i)

    def count(self, list):
        dict = {}
        for i in list:
            if i not in dict.keys():
                dict[i] = 0.001
            else:
                dict[i] += 1
        return dict

    def count_px(self):
        for i in range(len(self.__attributes)):
            temp = self.__trainSet[:,i]
            tdict = self.count(temp)
            self.__px.append(tdict)

    def count_pc(self):
        tag = self.__trainSet[:,-1]
        self.__pc = self.count(tag)

    def count_pxc(self):
        for i in range(len(self.__trainSet)):
            for j in range(len(self.__attributes)):
                self.__pxc[self.__trainSet[i][-1]][j][self.__trainSet[i][j]] += 1

    def train(self):
        self.count_px()
        self.count_pc()
        for i in self.__pc.keys():
            t = []
            for j in range(len(self.__attributes)):
                tdict = {}
                for k in self.__px[j].keys():
                    tdict[k] = 0.001
                t.append(tdict)
            self.__pxc[i] = t

        self.count_pxc()

    def predict(self, blist):
        result = {}
        for i in self.__pc.keys():
            pxc = 1
            px = 1
            for j in range(len(self.__attributes)):
                pxc *= self.__pxc[i][j][blist[j]] / self.__pc[i]
                px *= self.__px[j][blist[j]] / len(self.__trainSet)

            result[i] = pxc * self.__pc[i] / len(self.__trainSet) / px
        largest = max(result.values())
        for i in self.__pc.keys():
            if result[i] == largest:
                t = i
        return t

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0

    def getPrediction(self,testSet):
        predicti = []
        for i in testSet:
            result = self.predict(i)
            predicti.append(result)

        # print(result)
        accuracy = self.getAccuracy(testSet, predicti)
        return accuracy,predicti
        #print(accuracy)

'''

attr, train = tk.readDataSet("../dataset/transfusion.csv")

print(train[:,-1])

car_naive = Naive_bayes(attr,train)
print(car_naive.getPrediction(train))

#car_naive2 = Naive_bayes("./txtfile/car_prepro.txt", 0.5)
#car_naive2 = Naive_bayes("../car.txt", 0.5)
'''

