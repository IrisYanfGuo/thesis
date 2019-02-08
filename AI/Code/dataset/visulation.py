import toolkit as tk
import pandas as pd
from pandas.plotting import radviz,parallel_coordinates
import matplotlib.pyplot as plt

data = pd.read_csv("./iris.csv")
col = data.columns
print(col)

print(data.info())

'''

for i in col:
    data[i]=pd.Categorical.from_array(data[i]).codes


pd.scatter_matrix(data)

'''
#car = pd.read_csv("./car.csv")
#print(car)
#car.plot(kind='box')
#plt.show()

import seaborn as sns

sns.set()


sns.pairplot(data,hue="class")
plt.show()