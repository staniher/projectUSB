from flask import Flask, render_template, request

app=Flask(__name__)

import joblib
modele=joblib.load("modeles/meilleurModele.pkl")

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
	surface=float(request.form["surface"])
	rooms=int(request.form["rooms"])
	clim=int(request.form["clim"])
	etage=int(request.form["etage"])
	parking=int(request.form["parking"])
	import numpy as np
	features=[surface,rooms,clim,etage,parking]
	X_test=np.array([features])
	prediction=modele.predict(X_test)[0]
	return render_template("resultats.html", 
		predictions=round(prediction,2))

if __name__=="__main__":
	app.run(debug=True)