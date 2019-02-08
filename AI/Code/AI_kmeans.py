
import toolkit
import numpy as np

class kmeans(object):
    """docstring for kmeans"""
    # read trainSet, "attributes", and the number of cluster
    def __init__(self,attributes,trainSet):
        self.__predictions = []
        self.__k = 3
        self.__attributes = attributes
        self.__trainSet = trainSet
        #self.__training()
        


    def __getEuclideanDistance(self,data1,data2):
        return np.sqrt(np.sum(np.power(data1 - data2, 2)))

    #print(np.random.rank(k))
    def __training(self):
        nrow, ncol = np.shape(self.__trainSet)
        cluster_points = np.mat(np.zeros((self.__k,ncol)))
        for j in range(ncol):
            min_value = np.min(self.__trainSet[:,j])
            max_value = np.max(self.__trainSet[:,j])
            # find random cluster points, 
            # add row by row
            cluster_points[:,j] = min_value + (max_value-min_value)*np.random.rand(self.__k,1)
        print("Initial starting points (random):")
        for i in range(self.__k):
            print(cluster_points[i,:])
        
        clusterTable = np.mat(np.zeros((nrow,3))) #record index, dist, cluster
        for i in range(nrow):
            clusterTable[i,0] = i

        flag = True
        iteration_count = 0
        while flag:
            iteration_count += 1
            flag = False
            for i in range(nrow):
                min_dist = np.inf 
                min_index = -1
                for j in range(self.__k):
                    dist = self.__getEuclideanDistance(cluster_points[j,:],self.__trainSet[i,:])
                    if dist < min_dist:
                        min_dist = dist
                        min_index = j
                if clusterTable[i,1] != min_index:
                    flag = True
                    clusterTable[i,:] = i,min_index,np.power(min_dist,2)

            for kk in range(self.__k):
                new_cluster = np.mat(np.zeros((1,ncol)))
                count = 0
                for j in range(nrow):
                    if (clusterTable[j,1]) == kk:
                        count = count + 1
                        new_cluster = new_cluster + self.__trainSet[j,:]
                new_cluster = new_cluster / count
                for r in range(ncol):
                    cluster_points[kk,r] = new_cluster[0,r]

        print("Number of iterations: "+str(iteration_count))
        return cluster_points,clusterTable

    def getPrediction(self,k=3):
        self.__k = k
        cluster_points,clusterTable = self.__training()
        return cluster_points,clusterTable
        
