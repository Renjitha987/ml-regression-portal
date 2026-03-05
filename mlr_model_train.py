import pandas as pd
import pickle

# load dataset
dataset = pd.read_csv("datasets/cardata.csv")

# select important features
X = dataset[['Present_Price','Kms_Driven','Fuel_Type','Transmission']]
y = dataset['Selling_Price']

# encode categorical columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"), [2,3])
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

# train model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,y)

# save encoder + model
pickle.dump((ct,model), open("models/mlr_model.pkl","wb"))

print("MLR model saved successfully")