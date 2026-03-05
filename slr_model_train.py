# ===============================
# Import Libraries
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ===============================
# Load Dataset
# ===============================

dataset = pd.read_csv("fruits_dataset.csv")

# Independent variable
X = dataset.iloc[:, 1:2].values

# Dependent variable
y = dataset.iloc[:, 2:3].values


# ===============================
# Split Dataset
# ===============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# ===============================
# Train Model
# ===============================

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)


# ===============================
# Predictions
# ===============================

y_pred = regressor.predict(X_test)

print("Predicted Values:")
print(y_pred)

print("Actual Values:")
print(y_test)

print("Model Accuracy:", regressor.score(X, y))


# Example Prediction
predo = regressor.predict([[2]])

print("Prediction for input 2:", predo)


# ===============================
# Visualization
# ===============================

plt.scatter(X_train, y_train, color="green")
plt.plot(X_train, regressor.predict(X_train), color="purple")
plt.title("Calories (kcal) Vs Carbohydrates (g)")
plt.xlabel("Carbohydrates")
plt.ylabel("Calories")
plt.show()

plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Calories (kcal) Vs Carbohydrates (g)")
plt.xlabel("Carbohydrates (g)")
plt.ylabel("Calories")
plt.show()


# ===============================
# Save Model using Pickle
# ===============================

with open("models/slr_model.pkl", "wb") as file:
    pickle.dump(regressor, file)

print("SLR model saved successfully!")