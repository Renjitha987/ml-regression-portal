from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)

# model availability
models_available = {
    "slr": False,
    "mlr": False,
    "plr": False
}

# ---------------- LOAD MODELS ----------------

slr_model = None
mlr_model = None
plr_model = None
ct = None
poly = None

try:
    slr_model = pickle.load(open("models/slr_model.pkl","rb"))
    models_available["slr"] = True
except Exception as e:
    print("Error loading SLR model:", e)

try:
    ct, mlr_model = pickle.load(open("models/mlr_model.pkl","rb"))
    models_available["mlr"] = True
except Exception as e:
    print("Error loading MLR model:", e)

try:
    poly, plr_model = pickle.load(open("models/plr_model.pkl","rb"))
    models_available["plr"] = True
except Exception as e:
    print("Error loading PLR model:", e)

# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("home.html")


# ---------------- SLR ----------------

@app.route("/slr", methods=["GET","POST"])
def slr():

    if not models_available["slr"]:
        return "SLR Model Not Available"

    if request.method == "POST":

        x = float(request.form["x"])

        prediction = slr_model.predict([[x]])
        result = prediction[0][0]

        X = np.array([[1],[2],[3],[4],[5]])
        y = slr_model.predict(X)

        plt.scatter(X,y,color="blue")
        plt.plot(X,y,color="red")
        plt.scatter([x],[result],color="green",s=100)

        plt.title("Simple Linear Regression")
        plt.xlabel("Input")
        plt.ylabel("Output")

        plt.savefig("static/slr_graph.png")
        plt.close()

        return render_template("result.html", result=result, graph="slr_graph.png")

    return render_template("slr.html")


# ---------------- MLR ----------------

@app.route("/mlr", methods=["GET","POST"])
def mlr():

    if not models_available["mlr"]:
        return "MLR Model Not Available"

    if request.method == "POST":

        price = float(request.form["price"])
        kms = float(request.form["kms"])
        fuel = request.form["fuel"]
        trans = request.form["trans"]

        X = ct.transform([[price, kms, fuel, trans]])

        prediction = mlr_model.predict(X)

        result = prediction[0]

        return render_template("result.html", result=result)

    return render_template("mlr.html")


# ---------------- PLR ----------------

@app.route("/plr", methods=["GET","POST"])
def plr():

    if not models_available["plr"]:
        return "PLR Model Not Available"

    if request.method == "POST":

        x = float(request.form["x"])

        X_poly = poly.transform([[x]])
        prediction = plr_model.predict(X_poly)
        result = prediction[0]

        X_grid = np.arange(0,20,0.1)
        X_grid = X_grid.reshape(-1,1)

        X_poly_grid = poly.transform(X_grid)

        plt.scatter(X_grid, plr_model.predict(X_poly_grid), color="red")
        plt.plot(X_grid, plr_model.predict(X_poly_grid), color="blue")

        plt.scatter([x],[result], color="green", s=100)

        plt.xlabel("Present Price")
        plt.ylabel("Selling Price")
        plt.title("Polynomial Regression")

        plt.savefig("static/plr_graph.png")
        plt.close()

        return render_template("result.html", result=result, graph="plr_graph.png")

    return render_template("plr.html")


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)