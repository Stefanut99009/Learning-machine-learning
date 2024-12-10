from sklearn import linear_model
import pandas as pd

csv_data = pd.read_csv('ols_dataset.csv')

X = csv_data.iloc[:, :-1]
y = csv_data.iloc[:, -1]

reg = linear_model.LinearRegression()
reg.fit(X, y)

print("Linear Regression Model:", reg)
print("Coefficients:", reg.coef_)
