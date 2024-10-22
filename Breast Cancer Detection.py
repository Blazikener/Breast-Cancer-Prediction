# performing linear algebra
import numpy as np

# data processing
import pandas as pd

# visualisation
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data.head()

data.info()

#Removing unwanted data to increase performance speed
data.drop(['Unnamed: 32','id'],axis = 1,inplace = True)
data.diagnosis = [1 if each in data.diagnosis == 'M' else 0 for each in data.diagnosis]

data.head()

y = data.diagnosis.values
x_data = data.drop(['diagnosis'],axis = 1)

x = (x_data - np.min(x_data))/(np.max(data) - np.min(x_data))


