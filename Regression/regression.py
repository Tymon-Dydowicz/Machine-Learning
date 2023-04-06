import numpy as np
import seaborn as sns
import sklearn.datasets
import sklearn.model_selection
from sklearn.linear_model import Ridge

areas = [77.0, 50.0, 44.0, 65.0, 65.0, 52.0, 48.0, 45.0, 45.0, 36.0, 53.0, 43.0, 53.0, 27.0, 100.0, 52.0, 84.0, 77.0, 42.0, 75.0, 47.0, 52.0, 64.0, 42.0, 85.0, 30.0]
prices = [282, 250, 262, 399, 285, 308, 336, 277, 275, 219, 332, 262, 334, 185, 370, 329, 179, 265, 237, 330, 265, 345, 307, 270, 355, 179]
def predict(x, w1, w0):
    return np.multiply(x, w1) + w0
feature = [1, 2, 3]
expected = [10, 20, 30]
predicted = predict(feature, 10, 0)

assert (np.array(predicted) == np.array(expected)).all()

feature = [1, 2, 3]
expected = [11, 12, 13]
predicted = predict(np.array(feature), 1, 10)

assert (np.array(predicted) == np.array(expected)).all()

def train_linear_regression(x, y, w1=0.0, w0=0.0, learning_rate=0.01, max_iters=10000):
    for _ in range(max_iters):     
        prediction = predict(x, w1, w0)
        w1 -= learning_rate  *  np.sum(np.dot(np.subtract(prediction, y), x)) / len(prediction)
        w0 -= learning_rate * np.sum(np.subtract(prediction, y)) / len(prediction)
    return (w1, w0)

x = [1, 2, 3]
y = [16, 26, 36]
w1, w0 = train_linear_regression(x, y)
print(f"Model: y={w1}x+{w0}")
assert abs(w1 - 10) <= 0.01
assert abs(w0 - 6) <= 0.01

w1, w0 = train_linear_regression(areas, prices, learning_rate=0.0001, max_iters=100000)
print(f"Model: price = {w1} * area + {w0}")

predicted_prices = predict(areas, w1, w0)

sns.relplot(x=areas, y=prices)
sns.lineplot(x=areas, y=predicted_prices)

def mse(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.square(y_true-y_pred).mean()

assert mse([1, 2, 3], [2, 5, 7]) == ((1-2)**2 + (2-5)**2 + (3-7)**2)/3

dataset = sklearn.datasets.load_diabetes()
print(dataset.DESCR)

print(f"The first example")
print("x[0]", dataset.data[0])
print("y[0]", dataset.target[0])

X, test_X, y, test_y = sklearn.model_selection.train_test_split(dataset.data, dataset.target, random_state=42, test_size=.2)
train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(X, y, random_state=43, test_size=.125)
print("Training set (70% of 442):", train_X.shape, train_y.shape)
print("Validation set (10% of 442):", val_X.shape, val_y.shape)
print("Test set (20% of 442):", test_X.shape, test_y.shape)

regressor = Ridge()
regressor.fit(train_X, train_y)

pred = regressor.predict(train_X)
mse(train_y, pred)

pred = regressor.predict(val_X)
mse(val_y, pred)

regressor = Ridge(alpha=10)
regressor.fit(train_X, train_y)
print("MSE on training:", mse(train_y, regressor.predict(train_X)))
print("MSE on validation:", mse(val_y, regressor.predict(val_X)))

def alpha_optimization(train_X, train_y, val_X, val_y, iters) -> float:
    alpha = 1
    regressor = Ridge(alpha = alpha)
    regressor.fit(train_X, train_y)
    pred = regressor.predict(val_X)
    mem = mse(val_y, pred)
    memA = alpha

    for _ in range(iters):
        for search in [-0.01, 0.01]:
            alphaT = memA + search
            regressor = Ridge(alpha = alphaT)
            regressor.fit(train_X, train_y)
            pred = regressor.predict(val_X)
            meanSquaredError = mse(val_y, pred)

            if meanSquaredError < mem:
                memA = alphaT
                mem = meanSquaredError
                break

    return memA

domain = np.arange(0, 3, 0.01)
x_axis = [i for i in domain]
y_axis = []
for i in domain:
    regressor = Ridge(alpha = i)
    regressor.fit(train_X, train_y)
    y_axis.append(mse(val_y, regressor.predict(val_X)))
sns.lineplot(x = x_axis, y = y_axis)

best_alpha = alpha_optimization(train_X, train_y, val_X, val_y, 100)
print("The best alpha is", best_alpha)

regressor = Ridge(alpha=best_alpha)
regressor.fit(train_X, train_y)
print("MSE on training:", mse(train_y, regressor.predict(train_X)))
print("MSE on validation:", mse(val_y, regressor.predict(val_X)))
print("MSE on test:", mse(test_y, regressor.predict(test_X)))
