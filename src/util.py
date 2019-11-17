import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .config import *


def get_base_dataset():
    return pd.read_csv(os.path.join(DATA_CLEAN_PATH, 'data.csv'))


def get_curated_dataset():
    return pd.read_csv(os.path.join(DATA_CLEAN_PATH, 'ml-curated-data.csv'))


def get_subset(df, state, gender, age):
    return df.query(f"state == '{state}' & gender == '{gender}' & age == '{age}'")


def plot(df, state, gender, age, x="t", y="wage"):
    sub_df = get_subset(df, state, gender, age)
    y_values = sub_df[y].values
    increments = y_values[1:] / y_values[:-1] - 1
    sub_df.plot.scatter(x=x, y=y)
    plt.title("Wage level for {} between {} from {}".format(gender, age, state))
    plt.xlabel("Year")
    plt.ylabel("Wage")
    plt.show()
    plt.plot(sub_df[x][1:].values, increments)
    plt.title("Increments (%) for {} between {} from {}".format(gender, age, state))
    plt.xlabel("Year")
    plt.ylabel("%")
    plt.show()


def get_random_group(df):
    return {
        "state": random.choice(df.state.unique()),
        "age": random.choice(df.age.unique()),
        "gender": random.choice(df.gender.unique())
    }


def get_groups(df):
    return [
        {
            "gender":gender,
            "state": state,
            "age": age
        }
            for gender in df.gender.unique()
            for state in df.state.unique()
            for age in df.age.unique()
    ]

