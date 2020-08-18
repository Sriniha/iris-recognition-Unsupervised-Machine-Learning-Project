# Unsupervised-Machine-Learning-Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\91709\Downloads\Iris.csv")

df.head()

df.shape

df.describe

x = df.drop('Species',axis=1)

y = df['Species']

kmeans = KMeans(n_clusters=3)

kmeans.fit(x)

pred = kmeans.predict(x)

pred

pd.Series(pred).value_counts()

kmeans.score(x)

