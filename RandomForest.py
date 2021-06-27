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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

## Read data
train = pd.read_csv("training_set_VU_DM.csv")
test = pd.read_csv("test_set_VU_DM.csv")

## Use 1/4 of data for training
#train = train.sample(frac = 0.01)

## Replace na
train = train.fillna(train.mode().iloc[0])
test = test.fillna(test.mode().iloc[0])
#train.fillna(value=0.0, inplace=True)
#test.fillna(value=0.0, inplace=True)

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

train = train.drop(["date_time"], axis=1)
test = test.drop(["date_time"], axis=1)

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

#print(train.head())

### RF

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
           'comp8_rate_percent_diff', 'month', 'hour', 'dayofweek','position']]  ## Features

y = train['scores'] ## Labels

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

## Keep original dataframes
X_prescale = X
X_test_prescale = X_test


## Transform data for algorithm (Normalization)
#scaler = MinMaxScaler()

#X = scaler.fit_transform(X)
#X_test = scaler.transform(X_test)



# Split dataset into training set and test set
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.20,random_state=5)

X_train = X
y_train = y


## No transformation needed

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [50, 75, 100],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [5, 10],
    'n_estimators': [50, 75, 100]
}


#Create a Gaussian Classifier

clf=RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = clf, param_grid = param_grid,
                           cv = 3, n_jobs = 1, verbose = 2)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

# Fit the grid search to the data
grid_search.fit(X_train_s, y_train_s)

print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test_s, y_test_s)

base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(X_train_s, y_train_s)
base_accuracy = evaluate(base_model, X_test_s, y_test_s)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


#Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train, y_train)

clf.fit(X_train_s,y_train_s)

feature_imp = pd.Series(clf.feature_importances_,index=list(X_prescale.columns)).sort_values(ascending=False)

print(feature_imp)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

## Get prediction
#y_pred = clf.predict(X_test)

y_pred_s = clf.predict(X_test_s)

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
#cv_scores = cross_val_score(clf, X, y, cv=5)
#print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))

print("Accuracy:",metrics.accuracy_score(y_test_s, y_pred_s))