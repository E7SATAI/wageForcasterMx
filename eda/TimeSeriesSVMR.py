# import libraries
#%matplotlib inline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# imports
import os
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Enable python to import modules from parent directory
import sys
sys.path.append("..")

from src.config import *

# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

class TimeSeriesSVMR:

    def __init__(self, dfName, numTests, colName, resfilename):
		# Read CSV file located at data/clean path
        curated_data_path = os.path.join(DATA_CLEAN_PATH, dfName)
        self.df_curated = pd.read_csv(curated_data_path, sep = ',', header=[0], encoding="latin1")
        self.numTest = numTests
        self.colTarget = colName
        self.resultfilename = resfilename

    def get_random_params(self):
        return {
            "kernel": random.choice(["linear", "poly", "rbf", "sigmoid"]),
            "degree": random.choice(range(2,5)),
            "gamma": random.choice((range(10, 100))) / 100 ,
            "coef0": random.choice((range(10, 100))) / 100,
            "C": random.choice(range(1, 100)),
            "epsilon": random.choice((range(10, 50))) / 100
        }

    def get_rsme(self, param, target_col, features):
    
        model=SVR(**param)
        model.fit(features, target_col)
        estimates = model.predict(features)
        error = np.asmatrix(target_col.values - estimates)
        sme = (error * error.T / len(error)).tolist()[0][0]
        return np.sqrt(sme)

    def separateColumns(self, data, colname):
        # Separate columns in attributes(X) and labels(Y)
        # coloname = 'wage_increase'
        X = data.drop(colname, axis=1)
        y = data[colname]
        return X,y

    def splitDataSet(self, X, y, testSize=0.30):
        # split data in X,y/train,test subsets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize)
        return X_train, X_test, y_train, y_test

    def executeTests(self, y_train, X_train):
        result = []
        for i in range(self.numTest):
            param = self.get_random_params()
            rsme = self.get_rsme(param, y_train, X_train)
            param["rsme"] = rsme
            result.append( param)
        return result

    def saveResults(self, nameFile, listName):
        np.savetxt(nameFile, listName, delimiter=",", fmt='%s')

    def getBestSME(self, listName):
        list_rsmes = []
        for i in range(len(listName)):
            list_rsmes.append(listName[i].get("rsme"))
        minpos = list_rsmes.index(min(list_rsmes))
        return list_rsmes[minpos], minpos

    def runTest(self ):
        X, y = self.separateColumns(self.df_curated, self.colTarget)
        X_train, X_test, y_train, y_test = self.splitDataSet(X, y, 0.30)
        results = self.executeTests(y_train, X_train)
        best_rsme, pos = self.getBestSME(results)
        results.append({"Best RSME": best_rsme, "Position": pos, "Parameters": results[pos]})
        self.saveResults(self.resultfilename, results)
        return best_rsme, results[pos]

if __name__ == '__main__':
    #
    procTimeSeries = TimeSeriesSVMR("ml-curated-data.csv", 10, 'wage_increase', "result10.csv")

    print( procTimeSeries.runTest() )
