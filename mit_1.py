import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

my_list = list(range(1000000))
my_arr = np.array(range(1000000))

%time for i in range(10): my_list2 = my_list * 2
%time for i in range(10): my_arr2 = my_arr * 2

x = np.arange(-10, 10, 0.01)
y = x * x
y1 = 1 / (1 + np.power(np.e, -x))
y2 = np.power(np.e, -x) / (1 + np.power(np.e, -x))

plt.plot(x, y)
plt.show()

plt.plot(x, y1)
plt.show()

plt.plot(x, y2)
plt.show()

dataset = pd.read_excel('dataset/blood.xlsx')

y = dataset.iloc[:, -1].values
x = dataset.iloc[:, 1].values
z = dataset.iloc[0:4, 1:3].values

m = np.array([[1, 2],
              [3, 4]])

m * m
m @ m

s = pd.Series([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'])

d = pd.DataFrame({1 : [1, 2, 3, 4, 5],
                  2 : [1, 2, 3, 4, 5]})

from random import choice

throws = []

for i in range(1000000):
    one = choice(range(1, 7))
    two = choice(range(1, 7))
    total = one + two
    throws.append(total)

plt.hist(throws, bins = 30)
plt.show()

dataset = pd.read_csv('dataset/Data_Pre.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN',
              strategy = 'mean',
              axis = 0)

imp.fit(X[:, 0:2])
X[:, 0:2] = imp.transform(X[:, 0:2])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:, 2] = lab.fit_transform(X[:, 2])
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [2])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

























