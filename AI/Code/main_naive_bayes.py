from validation import cross,nice_print_model_info,draw_map
from naive_bayes import Naive_bayes
from toolkit import *
from naive_bayes_conti import Naive_bayes_conti

iris_attr, iris = readDataSet("./dataset/iris.csv")
acc, mcca, ka, map, t1, t2 = cross(Naive_bayes_conti, iris_attr, iris)

nice_print_model_info(acc, mcca, ka)
draw_map(map)

iris_attr, iris = readDataSet("./dataset/car.csv")
acc, mcca, ka, map, t1, t2 = cross(Naive_bayes, iris_attr, iris)

nice_print_model_info(acc, mcca, ka)
draw_map(map)
