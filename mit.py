import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from random import choice

throw = []

for i in range(1000000):
    one = choice(range(1, 7))
    two = choice(range(1, 7))
    total = one + two
    throw.append(total)

plt.hist(throw, bins = 30)
plt.show()


dataset = pd.read_csv('dataset/DemographicData.csv')

x = dataset.iloc[:, 2].values
y = dataset.iloc[:, 3].values
z = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
z = lab.fit_transform(z)
lab.classes_

plt.scatter(x[z == 0], y[z == 0], c = "r", label = "High Income")
plt.scatter(x[z == 1], y[z == 1], c = "g", label = "Low Income")
plt.scatter(x[z == 2], y[z == 2], c = "b", label = "Lower Middle Income")
plt.scatter(x[z == 3], y[z == 3], c = "y", label = "Upper Middle Income")
plt.legend()
plt.show()


























