import math
import operator
from random import random
from numpy import mean
from numpy import var
import toolkit as tk




# iris, continuous variables
# car, discrete variables
class Naive_bayes_conti(object):


    def __init__(self, attributes, trainset):

        self.__attributes = attributes
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
        dict['mu'] = mean(list)
        dict['sigma'] = var(list)

        return dict

    def count_px(self):
        for i in range(len(self.__attributes)):
            temp = self.__trainSet[:, i]
            tdict = self.count(temp)
            self.__px.append(tdict)

    def count2(self, list):
        dict = {}
        for i in list:
            if i not in dict.keys():
                dict[i] = 1
            else:
                dict[i] += 1

        return dict

    def count_pc(self):
        tag = self.__trainSet[:, -1]
        self.__pc = self.count2(tag)


    def count_pxc(self):
        for i in self.__pc.keys():
            t = []
            for j in range(len(self.__trainSet)):
                if self.__trainSet[j][-1] == i:
                    t.append(self.__trainSet[j])
            t2=[]
            for k in range(len(t[0])-1):
                t2.append(self.count([t[j][k] for j in range(len(t))]))

            self.__pxc[i] = t2



    def train(self):
        self.count_px()
        self.count_pc()
        self.count_pxc()


            # 初值为1, 防止不存在键的情况

        self.count_pxc()

    def _nor_distri(self,mu,sigma,x):

        return 1/(math.sqrt(2*math.pi)*sigma)*pow(math.e,-(x-mu)**2/(2*sigma**2))



    def predict(self, blist):
        result = {}
        for i in self.__pc.keys():
            pxc = 1
            px = 1
            for j in range(len(self.__attributes)):
                pxc *= self._nor_distri(self.__pxc[i][j]['mu'],self.__pxc[i][j]['sigma'],blist[j])
                px *= self._nor_distri(self.__px[j]['mu'],self.__px[j]['sigma'],blist[j])

            result[i] = pxc * self.__pc[i] / px

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
        # print(correct)
        return (correct / float(len(testSet))) * 100.0

    def getPrediction(self,testset):
        predictions = []
        for i in testset:
            result = self.predict(i)
            predictions.append(result)

        # print(result)
        accuracy = self.getAccuracy(testset, predictions)
        return accuracy, predictions

