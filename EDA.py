import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


## Read training data
data_training = pd.read_csv("training_set_VU_DM.csv")
data_test = pd.read_csv("test_set_VU_DM.csv")

## Print bottom 5 rows of dataframe
#print(data_training.tail())



## Shape of df
#print(data_training.shape)

## Columns of df
print(data_training.columns)
print(data_test.columns)

## Print top 5 rows of dataframe
#print(data_training.head())

## Print number of unique values for each column
print(data_training.nunique())
## Nr. of unique values for booking bool column
#print(data_training['booking_bool'].unique())

## Null values
print("Null values: \n", data_training.isnull().sum())

## Correlation

correlation = data_training.corr()
print(correlation)
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
plt.show()

sns.displot(data_training['booking_bool'])
plt.show()
sns.displot(data_training['click_bool'])
plt.show()
sns.displot(data_training['price_usd'])
plt.show()

sns.pairplot(data_training)
plt.show()

data_training = data_training.dropna(axis=1,how='any')
print(data_training.isnull().sum())
print(data_training.info())

correlation = data_training.corr()
print(correlation)

data_training[["promotion_flag","position"]].groupby("promotion_flag").sum().position.plot(kind="pie",shadow=True,autopct="%1.1f%%",radius=1.2,startangle=120)
plt.title("Promotion Graph")
plt.show()

data_training=data_training.drop(["srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","srch_destination_id"],axis=1)

data_test=data_test.dropna(axis=1,how="any")

print(data_test.columns)

data_test=data_test.drop(["srch_id","date_time","site_id","visitor_location_country_id","prop_country_id","prop_id","srch_destination_id"],axis=1)

print(data_test.columns)

data_training =data_training [['prop_starrating', 'prop_brand_bool', 'prop_location_score1',
             'prop_log_historical_price', 'price_usd', 'promotion_flag',
             'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
             'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
             'random_bool']]


## KMeans

data = data_training
n_cluster = range(1,11)

kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), scores,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

km = KMeans(n_clusters = 8)
kmean=km.fit(data_training)
y_kmeans = km.predict(data_training)

fig = plt.figure(1, figsize = (7, 7))

ax = Axes3D(fig, rect=[0, 0, 0.95, 1],
            elev = 48, azim = 134)

ax.scatter(data_training.iloc[:, 4:5],
           data_training.iloc[:, 7:8],
           data_training.iloc[:, 11:12],
           c = km.labels_.astype(np.float), edgecolor = 'm')

ax.set_xlabel('USD')
ax.set_ylabel('srch_booking_window')
ax.set_zlabel('srch_saturday_night_bool')

plt.title('K Means', fontsize = 10)

plt.show()

print(km.predict(data_test))