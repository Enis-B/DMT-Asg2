import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA


scaler = StandardScaler()
scaler1 = MinMaxScaler()



## Read training data
train = pd.read_csv("training_set_VU_DM.csv")
test = pd.read_csv("test_set_VU_DM.csv")

## Drop na
train = train.dropna(axis=1,how='any')
date_time = train['date_time']
train = train.drop(columns=['date_time'])

print(train.isnull().sum())

## Features

features = train.columns

print(features)

print(train.head())


## PCA

x = train.loc[:,features].values

x = scaler.fit_transform(x) ## normalizing features


print(x.shape)
print(np.mean(x),np.std(x))

print(x)

pca_train = PCA(n_components=3)
pca_train.fit_transform(x)
pca_trained = pca_train.fit_transform(x)

print((pd.DataFrame(pca_train.components_,columns=train.columns,index = ['PC-1','PC-2','PC-3'])).to_string())


pca_df = pd.DataFrame(data = pca_trained)

print(pca_df.tail())

print('Explained variation per principal component: {}'.format(pca_train.explained_variance_ratio_))

#plt.figure(figsize=(8,6))
#plt.scatter(pca_trained[:,0],pca_trained[:,1],c=train['booking_bool'],cmap='rainbow')
#plt.xlabel('First principal component')
#plt.ylabel('Second Principal Component')
#plt.show()
