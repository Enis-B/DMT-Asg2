import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline




## Read data

train = pd.read_csv("training_set_VU_DM.csv")
test = pd.read_csv("test_set_VU_DM.csv")

## Use 1/4 of data for training
#train = train.sample(frac = 0.25)

## Replace na

#train.fillna(train.mode().iloc[0])
#test.fillna(test.mode().iloc[0])

train.fillna(value=0.0, inplace=True)
test.fillna(value=0.0, inplace=True)

print(train.head())

## Drop na

#train = train.dropna(axis=1,how='any')
#test = test.dropna(axis=1,how='any')

## Drop id-like columns

#train = train.drop(['srch_id','site_id','visitor_location_country_id',
#                    'prop_country_id', 'prop_id','srch_destination_id'],axis=1)

def add_date_features(
        in_data, datetime_key="date_time", features=["month", "hour", "dayofweek"]
):
    dates = pd.to_datetime(in_data[datetime_key])
    for feature in features:
        if feature == "month":
            in_data["month"] = dates.dt.month
        elif feature == "dayofweek":
            in_data["dayofweek"] = dates.dt.dayofweek
        elif feature == "hour":
            in_data["hour"] = dates.dt.hour

    return in_data

train = add_date_features(train)
test = add_date_features(test)

## Additional preprocess
train = train.drop(["date_time"], axis=1)
test = test.drop(["date_time"], axis=1)

#print(train.dtypes)

#train = pd.get_dummies(train, columns=["month", "hour", "dayofweek"])
#test = pd.get_dummies(test, columns=["month", "hour", "dayofweek"])


scores = []

for i, row in train.iterrows():
    if row['booking_bool'] == 1:
        scores.append(5)
    elif row['booking_bool'] == 0 and row['click_bool'] == 1:
        scores.append(1)
    elif row['booking_bool'] == 0 and row['click_bool'] == 0:
        scores.append(0)

train['scores'] = scores

#train = train.drop(["booking_bool","click_bool"], axis=1)

#print(train.head())
#print(train.isnull().sum())

## Features

features = train.columns
#print(features)


## DL

X = train[['srch_id', 'site_id', 'visitor_location_country_id',
           'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
           'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
           'prop_location_score1', 'prop_location_score2',
           'prop_log_historical_price', 'price_usd', 'promotion_flag',
           'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
           'srch_adults_count', 'srch_children_count', 'srch_room_count',
           'srch_saturday_night_bool', 'srch_query_affinity_score',
           'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv',
           'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
           'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
           'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
           'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
           'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
           'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
           'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
           'comp8_rate_percent_diff', 'month', 'hour', 'dayofweek']]

y = train['scores']

X_test = test[['srch_id', 'site_id', 'visitor_location_country_id',
               'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
               'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
               'prop_location_score1', 'prop_location_score2',
               'prop_log_historical_price', 'price_usd', 'promotion_flag',
               'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
               'srch_adults_count', 'srch_children_count', 'srch_room_count',
               'srch_saturday_night_bool', 'srch_query_affinity_score',
               'orig_destination_distance', 'random_bool', 'comp1_rate', 'comp1_inv',
               'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
               'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
               'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
               'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
               'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
               'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
               'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
               'comp8_rate_percent_diff', 'month', 'hour', 'dayofweek']]

## Keep original
X_prescale = X
X_test_prescale = X_test

## Transform data for algorithm (Normalization)
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

## Transform data for algorithm (Standardization)
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

## NN model

model = keras.Sequential()

model.add(Dense(105, input_dim=52, activation='relu'))
model.add(Dense(105, activation='relu'))
model.add(Dense(105, activation='relu'))
model.add(Dense(105, activation='relu'))
model.add(Dense(105, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['Accuracy'])

## Params and fit
model.fit(X, y, epochs=1, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

y_pred = model.predict(X_test)
## Print
print(y_pred)
y_joined = [j for i in y_pred for j in i]
print(y_joined)


#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y, y_joined))

print("Confusion Matrix:")
print(confusion_matrix(y, y_joined))

print("Classification Report")
print(classification_report(y, y_joined))

## Save to df
X_test_prescale['y_pred'] = y_joined

## Create result
result = X_test_prescale[['srch_id','prop_id','y_pred']]

result = result.sort_values(['srch_id','y_pred'], ascending=(True,False))

## Create submission
submission = result[['srch_id', 'prop_id']]

submission.to_csv('submission.csv',index=False)

