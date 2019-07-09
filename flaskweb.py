from flask import Flask, render_template, request
import pickle
from random_forest_manual import random_forest_predictions, decision_tree_algorithm, predict_example

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/result")
def result():
    # retrieve request data
    data = request.args

    forest = readForestData('forest.pkl')

    # example = [-1,1,1,-1,-1,-1,1,-1,-1,-1,0,1,1,-1,-1,-1,-1]
    # tree = forest[0]
    # tree = decision_tree_algorithm(train_df, max_depth=5)

    # prediction = predict_example(example, tree)
    
    # prediction = random_forest_predictions(
    #     [-1,1,1,-1,-1,-1,1,-1,-1,-1,0,1,1,-1,-1,-1,-1],
    #     forest
    # )
    
    # TODO: testing single data with forest data taken from pickle file

    return render_template('result.html', data=data, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

def readForestData(fileLocation):
    with open(fileLocation, 'rb') as f:
        try:
            data = list(f)
        except EOFError:
            data = list()

    return data
