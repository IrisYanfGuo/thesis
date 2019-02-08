import pandas as pd

iris = pd.read_csv("iris.csv")
#iris_fill1 = iris.fillna(1)
iris_bfill = iris.fillna(method="bfill")
#print(iris_fill1)
#print(iris_bfill)

c=iris_bfill.get_values()

# 取值的最后一列
print(type(c[:,-1]))
print(type(iris_bfill['variety']))

# 如何把类别型转化为数值
d = iris_bfill['variety'].astype('category')
e= d.cat.rename_categories([1,2,3])

#
print(type(e))

# 有用的查看数据类型的函数

#print(iris_bfill.info())

print(type(iris_bfill['sepal.length'][0]))

print(type(iris_bfill.dtypes))