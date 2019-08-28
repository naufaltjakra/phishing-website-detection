from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle
from random_forest_manual import random_forest_predictions

from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')


@app.route("/result", methods=['POST'])
def result():

    # Load model pickle
    rf_model = open("model.pkl", "rb")
    clf = joblib.load(rf_model)

    if request.method == 'POST':
        ip = request.form['ip']
        ul = request.form['ul']
        at = request.form['at']
        ps = request.form['ps']
        sd = request.form['sd']
        ht = request.form['ht']
        ru = request.form['ru']
        ua = request.form['ua']
        sfh = request.form['sfh']
        ab = request.form['ab']
        re = request.form['re']
        mo = request.form['mo']
        po = request.form['po']
        ad = request.form['ad']
        dns = request.form['dns']
        wt = request.form['wt']

        data = {'ip': [ip],
                'ul': [ul],
                'at': [at],
                'ps': [ps],
                'sd': [sd],
                'ht': [ht],
                'ru': [ru],
                'ua': [ua],
                'sfh': [sfh],
                'ab': [ab],
                're': [re],
                'mo': [mo],
                'po': [po],
                'ad': [ad],
                'dns': [dns],
                'wt': [wt]}

        # this cause KeyError
        df_data = pd.DataFrame(data, index=['row'])

        print(" ")
        print("data : ")
        print(data)

        print(" ")
        print("df_data : ")
        print(df_data)
        print(" ")

        prediction = random_forest_predictions(df_data, clf)

    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
