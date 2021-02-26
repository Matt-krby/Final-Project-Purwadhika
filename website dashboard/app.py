from flask import Flask, render_template, send_file, request
import os
import io
import csv
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import Normalizer
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import tree
from tabulate import tabulate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

app= Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def convert_dummy(df, feature,rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    df.drop([feature],axis=1,inplace=True)
    df=df.join(pos)
    return df

def RegEvaluation(model, ytest, xtest, nameindex, yname, totaldt):
    ypred = model.predict(xtest)
    xtest['Pred_Y'] = model.predict(xtest)
    dt = pd.merge(totaldt,xtest,how = 'right')
    xtest.drop(['Pred_Y'],axis=1,inplace=True)
    dt = dt[[nameindex, yname,'Pred_Y']]
    dt.sort_values(by = yname, ascending = False,inplace=True)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    r2 = r2_score(ytest, ypred)

@app.route('/', methods=['GET','POST'])
def home():
    salary_table = pd.read_csv("./dataset/season_salary.csv",encoding = 'utf-8')
    seasons = pd.read_csv("./dataset/season_status.csv",encoding = 'utf-8')

    seasons[seasons['Year']==2017]

    salary_table = salary_table[['Player','season17_18']] # hapus column team
    salary_table.rename(columns={'season17_18':'salary17_18'},inplace = True) # rename
    salary_table['salary17_18'] = salary_table['salary17_18']/1000000 # ubah jumlah ke juta usd

    seasons = seasons[seasons['Year']==2017] # ambil hanya data season 2017-2018
    stats17 = seasons[['Year','Player','Pos','Age','G','PER','MP','PTS','AST','TRB','TOV','BLK','STL']] # pisahin sehingga hanya data ini aja yang diambil

    stats17.drop_duplicates(subset=['Player'], keep='first',inplace=True) # hapus data pemain duplicate, keep yang pertama

    c = ['MPG','PPG','APG','RPG','TOPG','BPG','SPG']
    w = ['MP','PTS','AST','TRB','TOV','BLK','STL']

    for i,s in zip(c,w):
        stats17[i] = stats17[s] / stats17['G']

    stats17.drop(w,axis=1,inplace=True)
    stats17.loc[stats17['Pos'] == 'PF-C','Pos'] = 'PF'
    dataset_salary = pd.merge(stats17, salary_table) # simpan data untuk digunakan di home.html
    stats_salary = pd.merge(stats17, salary_table)

    stats_salary.drop_duplicates(subset = ["Player"], keep = "first", inplace = True)
    stats_salary.sort_values(by = "PPG", ascending = False, inplace = True)
    stats_salary.sort_values(by = "PER", ascending = False, inplace = True)
    stats_salary.sort_values(by='Age',ascending = False,inplace = True)
    stats_salary.sort_values(by='TOPG',ascending=False,inplace = True)

    sns.set(rc={'figure.figsize':(10,7.5)})

    pd.get_dummies(stats_salary["Pos"], prefix="Pos")

    stats_salary = convert_dummy(stats_salary,'Pos')

    stats_salary = stats_salary.dropna()
    Y = stats_salary['salary17_18']
    X = stats_salary.drop(['salary17_18','Year', 'Player'],axis=1)

    transformer = MaxAbsScaler().fit(X) # Scale each feature by its maximum absolute value.
    newX = transformer.transform(X)
    newX = pd.DataFrame(newX,columns = X.columns)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    x_train_NEW, x_test_NEW, y_train_NEW, y_test_NEW = train_test_split(newX, Y, test_size = 0.3)

    clf = tree.DecisionTreeRegressor(max_depth=4, criterion="mse")
    dtree = clf.fit(x_train, y_train)

    RegEvaluation(dtree, y_test, x_test, 'Player', 'salary17_18',stats_salary)

    dtree = clf.fit(x_train_NEW, y_train_NEW)
    RegEvaluation(dtree, y_test_NEW, x_test_NEW, 'Player', 'salary17_18', stats_salary)

    if request.method == 'POST': # jika dia melakukan submit form dengan method post, maka
        data = {
            "Age": float(request.form['age']), # ambil data request form dari input name age
            "G": float(request.form['g']), # ambil data request form dari input name g
            "PER": float(request.form['per']),
            "MPG": float(request.form['mpg']),
            "PPG": float(request.form['ppg']),
            "APG": float(request.form['apg']),
            "RPG": float(request.form['rpg']),
            "TOPG": float(request.form['topg']),
            "BPG": float(request.form['bpg']),
            "SPG": float(request.form['spg']),
            "Pos_C": 1 if request.form['pos'] == 'C' else 0, # jika dia selectnya adalah C, makan return nya 1 jika tidak 0
            "Pos_PF": 1 if request.form['pos'] == 'PF' else 0,
            "Pos_PG": 1 if request.form['pos'] == 'PG' else 0,
            "Pos_SF": 1 if request.form['pos'] == 'SF' else 0,
            "Pos_SG": 1 if request.form['pos'] == 'SG' else 0,
        }
        data_df = pd.DataFrame([data], columns = ['Age', 'G', 'PER', 'MPG', 'PPG', 'APG', 'RPG', 'TOPG', 'BPG', 'SPG', 'Pos_C', 'Pos_PF', 'Pos_PG', 'Pos_SF', 'Pos_SG'])
        salary_prediction = clf.predict(data_df) # buat variable salary_prediction, lalu dimasukkan ke html
    else:
        salary_prediction = [0] # ini defaultnya pake array 0, karena data dari salary_prediction adalah berupa array, maka defaultnya juga harus array

    return render_template('home.html', dataset_salary=dataset_salary, salary_prediction=salary_prediction) # masukkan data_salary dan salary_prediction agar bisa diakses di html nya

if __name__ == '__main__':
    app.run(debug=True)
