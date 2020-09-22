from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

# feature and labels
X = boston.data
y = boston.target

# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

l_reg = linear_model.LinearRegression()

# ploting
plt.scatter(X.T[5], y)
plt.show()

# model creation
model = l_reg.fit(X_train, y_train)

predictions = model.predict(X_test)

print('Predictions: ', predictions)

print('R^2: ', l_reg.score(X,y))
print('coeff: ', l_reg.coef_)
print('intercept: ', l_reg.intercept_)