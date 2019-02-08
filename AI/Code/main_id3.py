#!/usr/local/bin/python3
import toolkit as tk 
from AI_id3 import id3
import numpy as np
from validation import cross, nice_print_model_info,draw_map


attributes,instances = tk.readDataSet("./dataset/car.csv")
testSet = tk.readDataSet("./dataset/car.csv")[1]
#trainSet,testSet = tk.splitDataSet(instances,0.9)


attributes = (list(attributes))

instances = instances.tolist()
testSet = testSet.tolist()
#trainSet = trainSet.tolist()
#print(len(testSet))

myid3 = id3(attributes,instances)
#acc,pre = myid3.getPrediction(testSet)
#print(acc,pre)
#print(myid3.getTree())

tk.createPlot(myid3.getTree())

