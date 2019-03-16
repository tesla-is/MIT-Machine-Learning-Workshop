import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('dataset/blood.xlsx')
X = dataset.iloc[: , 1].values
y = dataset.iloc[: , -1].values
X = X.reshape(-1, 1)

plt.scatter(X, y)
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure')
plt.title('Relationship between Age and Blood Pressure')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

plt.scatter(X_train, y_train)
plt.plot(X_train, lin_reg.predict(X_train), c = "r")
plt.xlabel('Age')
plt.ylabel('Systolic 0Blood Pressure')
plt.title('Relationship between Age and Blood Pressure (Training Set)')
plt.show()

plt.scatter(X_test, y_test)
plt.plot(X_test, lin_reg.predict(X_test), c = "r")
plt.xlabel('Age')
plt.ylabel('Systolic 0Blood Pressure')
plt.title('Relationship between Age and Blood Pressure (Test Set)')
plt.show()


y_pred = lin_reg.predict(X_test)










