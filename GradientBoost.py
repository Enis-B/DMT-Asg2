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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

## Read data

train = pd.read_csv("training_set_VU_DM.csv")
test = pd.read_csv("test_set_VU_DM.csv")

## Use 1/4 of data for training
train = train.sample(frac = 0.25)

## Replace na

#train.fillna(train.mode().iloc[0])
#test.fillna(test.mode().iloc[0])

train.fillna(value=0.0, inplace=True)
test.fillna(value=0.0, inplace=True)

#print(train.head())
#print(train.shape)
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


## random_bool, click_bool,booking_bool,position

## Models

X = train[['srch_id', 'site_id', 'visitor_location_country_id',
           'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
           'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
           'prop_location_score1', 'prop_location_score2',
           'prop_log_historical_price', 'price_usd', 'promotion_flag',
           'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
           'srch_adults_count', 'srch_children_count', 'srch_room_count',
           'srch_saturday_night_bool', 'srch_query_affinity_score',
           'orig_destination_distance','random_bool','comp1_rate', 'comp1_inv',
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
           'orig_destination_distance','random_bool', 'comp1_rate', 'comp1_inv',
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

## Transform data for algorithm
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)



## Split for evaluation testing
state = 5
test_size = 0.3

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=state)

## Create GBT Classifier, LGMClassifier, XGBClassifier

gb_clf = GradientBoostingClassifier()
gb_rg = GradientBoostingRegressor()

lg_clf = lightgbm.LGBMClassifier(objective='multiclass', num_class = 3)
lg_rnk = lightgbm.LGBMRanker(objective="lambdarank",
                             metric="ndcg")

xgb_clf = XGBClassifier(objective='multi:softmax', num_class = 3)

group = [len(train)/3,len(train)/3,len(train)/3]

## Fit model
#gb_clf.fit(X_train, y_train)
#lg_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)
#lg_clf.fit(X,y)
#lg_rnk.fit(X,y,group=group)


## Important features visualization
feature_imp = pd.Series(xgb_clf.feature_importances_,index=list(X_prescale.columns)).sort_values(ascending=False)
#feature_imp = pd.Series(gb_clf.feature_importances_,index=list(X_prescale.columns)).sort_values(ascending=False)
#feature_imp = pd.Series(lg_clf.feature_importances_,index=list(X_prescale.columns)).sort_values(ascending=False)
#print(feature_imp)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


## Get prediction
#y_pred = gb_clf.predict(X_test)
#y_pred = xgb_clf.predict(X_test)
#y_pred = lg_clf.predict(X_test)
#y_pred = lg_rnk.predict(X_test)

#y_pred_val = gb_clf.predict(X_val)
y_pred_val = xgb_clf.predict(X_val)
#y_pred_val = lg_clf.predict(X_val)


## Save to df
#X_test_prescale['y_pred'] = y_pred
## Print
#print(y_pred)

## Create result
#result = X_test_prescale[['srch_id','prop_id','y_pred']]

#result = result.sort_values(['srch_id','y_pred'], ascending=(True,False))

## Create submission
#submission = result[['srch_id', 'prop_id']]

#submission.to_csv('submission.csv',index=False)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_val, y_pred_val))

#cv_scores = cross_val_score(gb_clf, X, y, cv=5)
#print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))

print("Classification Report")
print(classification_report(y_val, y_pred_val))

print("Accuracy score (training): {0:.3f}".format(xgb_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(xgb_clf.score(X_val, y_val)))


#print("Accuracy score (training): {0:.3f}".format(xgb_clf.score(X_train, y_train)))
#print("Accuracy score (validation): {0:.3f}".format(xgb_clf.score(X_val, y_val)))
#score = xgb_clf.score(X_val, y_val)
#print(score)#print("Accuracy score (training): {0:.3f}".format(lg_clf.score(X_train, y_train)))
#print("Accuracy score (validation): {0:.3f}".format(lg_clf.score(X_val, y_val)))
#score = lg_clf.score(X_val, y_val)
#print(score)
