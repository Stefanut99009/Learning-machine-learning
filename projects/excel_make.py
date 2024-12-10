import numpy as np
import pandas as pd

np.random.seed(42)

num_samples = 100

num_features = np.random.randint(3, 6)
X = np.random.rand(num_samples, num_features)

coefficients = np.random.rand(num_features)

noise = np.random.normal(0, 0.1, num_samples)
y = X.dot(coefficients) + noise

data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(num_features)])
data['Target'] = y

print(data.head())

data.to_csv("ols_dataset.csv", index=False)
