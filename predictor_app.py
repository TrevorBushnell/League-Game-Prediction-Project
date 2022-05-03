import os
import pickle
from flask import Flask, jsonify, request
from sklearn.tree import DecisionTreeClassifier 
import mysklearn.myutils
import mysklearn.myutils as myutils


import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 


import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

import mysklearn.myevaluation
import mysklearn.myevaluation as myevaluation

import copy
import numpy as np

# now we create a flask app
# by default flask run on port 5000
app = Flask(__name__)

# now we need to define "routes"
# a route is a function that handles a request
# lets have a route for the root index page
@app.route("/", methods=["GET"])
def index():
    # we can return content and status code
    return "<h1>Welcome to Ben Lombardi and Trevor Bushnell's League of Legends Prediction app!!</h1><br><h1>The attribute domains are : <br> {'blueEliteMonsters': ['-1.0 - 0.0', '0.0 - 1.0', '1.0 - 2.0'], 'blueDragons': ['0.0', '1.0'], 'blueGoldDiff': ['-9152.0 - -7360.0', '1608.0 - 3400.0', '3400.0 - 5192.0', '5192.0 - 6984.0', '6984.0 - 8776.0'], 'blueExperienceDiff': ['-8290.0 - -6627.0', '1696.0 - 3359.0', '3359.0 - 5022.0', '5022.0 - 6685.0', '6685.0 - 8348.0']}<h1>", 200

# now the /predict route
@app.route("/predict", methods=["GET"])
def predict():
    # we need to parse the unseen instance's
    # attribute values from the request
    monsters = request.args.get("blueEliteMonsters", "") # "" default value
    dragons = request.args.get("blueDragons", "")
    gold_diff = request.args.get("blueGoldDiff", "")
    exp_diff = request.args.get("blueExperienceDiff", "")
    print("level:", monsters, dragons, gold_diff, exp_diff)

    # TODO: fix the hardcoding
    prediction = predict_winner([monsters,
       dragons, gold_diff, exp_diff])
    # if anything goes wrong, predict_interviewed_well()
    # is going to return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def predict_winner(instance):
    # we need to traverse the interview tree
    # and make a prediction for instan
    # how do we get the tree here?
    # generally, we need to save a trained
    # ML mode from another process for use later
    # (in this process)
    # enter pickling
    # unpickle tree.p
    df = MyPyTable()
    df.load_from_file("high_diamond_ranked_10min.csv")
    y = df.get_column("blueWins")

    X = copy.deepcopy(df.data)
    # randomizing indexes to sort out a stratified sample
    X_indexes = [ i for i in range(len(X))]
    myutils.randomize_in_place(X_indexes,y,0)
    grouped_X =[]
    grouped_y = []
    for i in range(len(X)):
        if grouped_y.count(y[i]) >0:
            grouped_X[grouped_y.index(y[i])].append(X_indexes[i])
        else:
            grouped_y.append(y[i])
            grouped_X.append([X_indexes[i]])
    X_data = []
    y_data = []
    for i in range(1500):
        for j in range(len(grouped_X)):
            X_data.append(X[grouped_X[j][i]])
            y_data.append(grouped_y[j])

    X = copy.deepcopy(X_data)

    y = copy.deepcopy(y_data)
    for entry in X:
        del entry[0:8]
        del entry[2:9]
        del entry[4:]
    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X,y)
    print(instance)
    # prediction time!!
    try:
        prediction = tree_clf.predict(instance)
        return prediction
    except:
        print("error")
        return None




if __name__ == "__main__":
    # deployment notes
    # we need to get our web app on the web
    # we can setup/maintain our own server OR
    # we can use a cloud provider
    # lots of cloud providers: AWS, GCP, Azure, DigitalOcean, 
    # Heroku, ...
    # we are going to use Heroku (PaaS platform as a service)
    # lots of ways to deploy a Flask app to Heroku
    # see my youtubes videos for 4 different ways
    # we will do what I call 2.B.
    # deploying a docker container using heroku.yml and git
    # first, we need to change some app settings
    # heroku is going to set the port for our app
    # to use via an enviroonment variable
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, port=port, host="127.0.0.1") # TODO: turn off debug
    # when you deploy to production