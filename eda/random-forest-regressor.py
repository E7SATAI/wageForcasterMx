#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from scipy import stats
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


# In[3]:


import sys
sys.path.append("..")

from src.config import *


# ## Import clean data

# In[4]:


# Read data
data_path = os.path.join(DATA_CLEAN_PATH, "ml-curated-data.csv")
dfCurated = pd.read_csv(data_path)
dfCurated.head()


# In[5]:


target_col = "wage_increase"
features = [c for c in dfCurated.columns if c != target_col]

train = dfCurated.sample(frac=0.7)
test = dfCurated.drop(train.index)


# In[6]:


train_x = train.drop(target_col, 1)
train_y = train.drop(features, 1)

test_x = test.drop(target_col, 1)
test_y = test.drop(features, 1)


# In[7]:


regr = RandomForestRegressor(max_depth=1, n_estimators=5000, warm_start=True, max_features="sqrt", min_impurity_decrease=0.1)
regr.fit(train_x, np.ravel(train_y)) 


# In[8]:


estimates = regr.predict(train_x)
error = np.asmatrix(train_y.values - estimates)
sme = (error.T * error / len(error)).tolist()[0][0]
sme


# In[9]:


np.sqrt(sme)


# In[10]:


def get_random_params():
    return {
        "n_estimators": random.choice((range(1, 900))),
        "criterion": random.choice(["mse", "mae"]),
        "max_depth": random.choice(list(range(1, 100)) + [None]),
        "random_state": random.choice((range(10, 100))),
        "max_features": random.choice(range(10, 100)) / 100,
        "min_impurity_decrease": random.choice((range(10, 100)))/100,
    }

param = get_random_params()
param


# In[11]:


def get_rsme(df, param, target_col, features):
    train = df.sample(frac=0.7)
    test = df.drop(train.index)
    train_x = train.drop(target_col, 1)
    train_y = train.drop(features, 1)
    test_x = test.drop(target_col, 1)
    test_y = test.drop(features, 1)
    model= RandomForestRegressor(**param)
    model.fit(train_x, np.ravel(train_y))
    estimates = model.predict(train_x)
    error = np.asmatrix(train_y.values - estimates)
    sme = (error.T * error / len(error)).tolist()[0][0]
    return np.sqrt(sme)


# In[ ]:


get_rsme(dfCurated, param, target_col="wage_increase", features=[c for c in dfCurated.columns if c != "wage_increase"])


# In[ ]:


result = []
for i in range(2000):
    param = get_random_params()
    rsme = get_rsme(dfCurated, param, target_col="wage_increase", features=[c for c in dfCurated.columns if c != "wage_increase"])
    param["rsme"] = rsme
    result.append(param)


# In[ ]:


result_df = pd.DataFrame(result)
result_df.head()


# In[ ]:


output_path = os.path.join(DATA_CLEAN_PATH, "param_random_forest_2.csv")
result_df.to_csv(output_path)


# In[ ]:


result_df.max_depth.unique()
result_df.describe()


# In[ ]:


result_df.rsme.min()


# In[ ]:




