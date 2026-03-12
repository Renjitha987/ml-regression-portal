import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# -------------------------------
# Step 1: Load Dataset
# -------------------------------

dataset = pd.read_csv("datasets/cardata.csv")

print("Dataset loaded successfully")
print(dataset.head())


# -------------------------------
# Step 2: Define Variables
# -------------------------------

X = dataset.iloc[:, 3:4].values   # Present_Price
y = dataset.iloc[:, 2].values     # Selling_Price


# -------------------------------
# Step 3: Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# Step 4: Polynomial Transformation
# -------------------------------

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# -------------------------------
# Step 5: Train Model
# -------------------------------

model = LinearRegression()
model.fit(X_train_poly, y_train)


# -------------------------------
# Step 6: Prediction
# -------------------------------

y_pred = model.predict(X_test_poly)


# -------------------------------
# Step 7: Model Evaluation
# -------------------------------

print("R2 Score:", r2_score(y_test, y_pred))



# -------------------------------
# Step 8: Visualization
# -------------------------------

X_grid = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

X_grid_poly = poly.transform(X_grid)

y_curve = model.predict(X_grid_poly)

plt.figure(figsize=(8, 6))
plt.scatter(X.flatten(), y, color="red", alpha=0.8, label="Actual Data")
plt.plot(X_grid.flatten(), y_curve, color="blue", linewidth=2, label="Polynomial Fit")

plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Polynomial Regression")
plt.legend()

plt.show()


# -------------------------------
# Step 9: Save Model
# -------------------------------

pickle.dump((poly, model), open("models/plr_model.pkl", "wb"))

print("Polynomial Regression model saved successfully!")