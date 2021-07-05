import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

## Read data
train = pd.read_csv("training_set_VU_DM.csv")
test = pd.read_csv("test_set_VU_DM.csv")

## Use 1/4 of data for training
#train = train.sample(frac = 0.01)

## Replace na
train = train.fillna(train.mode().iloc[0])
test = test.fillna(test.mode().iloc[0])

#train.fillna(value=0.0, inplace=True)

## Drop na
#train = train.dropna(axis=1,how='any')

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

#train['mean_prop_score'] = ''
#train['mean_prop_score'] = np.NAN

#for i, row in train.iterrows():
#    row['mean_prop_score'] = int((row['prop_location_score1'] + row['prop_location_score2']) / 2)

#print(train['mean_prop_score'].head())

#test['mean_prop_score'] = ''
#test['mean_prop_score'] = np.NAN

#for i, row in test.iterrows():
#    row['mean_prop_score'] = int((row['prop_location_score1'] + row['prop_location_score2']) / 2)

#print(test['mean_prop_score'].head())


#ratings_dict = {
#    "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
#    "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
#    "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3], }

# To use item-based cosine similarity
#sim_options = {
#    "name": "cosine",
#    "user_based": False,  } # Compute  similarities between items

#algo = KNNWithMeans(sim_options=sim_options)

#reader = Reader(rating_scale=(0, 5))

def add_scores(train):
    scores = []
    for i, row in train.iterrows():
        if row['booking_bool'] == 1:
            scores.append(5)
        elif row['booking_bool'] == 0 and row['click_bool'] == 1:
            scores.append(1)
        elif row['booking_bool'] == 0 and row['click_bool'] == 0:
            scores.append(0)
    train['scores'] = scores
    return train

train = add_scores(train)

y = train['scores']

train = train.drop(["booking_bool","click_bool","scores"], axis=1)

#print(train.head())

#print(train.isnull().sum())

## Features

#features = train.columns
#print(train.describe())
#print(train.dtypes)

#print(features)

#print(train.head())

def add_mean_features(train):
    stich = train.groupby('prop_id',as_index=False).mean()
    #print(stich.head())
    train = pd.merge(left=train,right=stich, how='left',left_on='prop_id',right_on='prop_id'
                     , suffixes= ('', '_mean'))
    #print(train.columns)
    #print(train.head())
    print(train.shape)
    return train

def add_std_features(train):
    stich = train.groupby('prop_id',as_index=False).std()
    #print(stich.head())
    train = pd.merge(left=train,right=stich, how='left',left_on='prop_id',right_on='prop_id'
                     , suffixes= ('', '_std'))
    #print(train.columns)
    #print(train.head())
    print(train.shape)
    return train

def add_median_features(train):
    stich = train.groupby('prop_id',as_index=False).median()
    #print(stich.head())
    train = pd.merge(left=train,right=stich, how='left',left_on='prop_id',right_on='prop_id'
                     , suffixes= ('', '_median'))
    #print(train.columns)
    #print(train.head())
    print(train.shape)
    return train


train = add_mean_features(train)

#train1 = add_std_features(train)
#train2 = add_median_features(train)

#train = pd.merge(left=train,right=train1)

#print(train.shape)

#train = pd.merge(left = train, right = train2)

#print(train.shape)

test = add_mean_features(test)

#est1 = add_std_features(test)
#test2 = add_median_features(test)

#test = pd.merge(left=test,right=test1)

#print(test.shape)

#test = pd.merge(left = test, right = test2)

#print(test.shape)


train = add_date_features(train)
test = add_date_features(test)

train = train.drop(["date_time","comp6_inv_mean","comp7_inv_mean","comp8_inv_mean"], axis=1)
test = test.drop(["date_time"], axis=1)


print("Train shape:\n",train.shape)
print("Train columns:\n",train.columns)

print("Test shape:\n",test.shape)
print("Test columns:\n",test.columns)



### RF

'''
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
           'comp8_rate_percent_diff','position']]  ## Features # position, day ,month, year

'''

'''
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
               'comp8_rate_percent_diff']]

'''
## Keep original dataframe (commented to save memory)
#X_prescale = X

## Transform data for algorithm (Normalization)
#scaler = MinMaxScaler()

#X = scaler.fit_transform(X)

# Split dataset into training set and test set
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(train, y, test_size=0.30, random_state=1997)

#X_train = X
#y_train = y

## testing recommender !

#new_train = pd.DataFrame(X_train_s)

#print(new_train.head())

#new_train = new_train[["srch_id", "prop_id"]]

#new_train['scores'] = list(y_train_s)

#print(new_train.head())

#data_rec = Dataset.load_from_df(new_train[["srch_id", "prop_id","scores"]], reader)
#trainingSet = data_rec.build_full_trainset()

#algo.fit(trainingSet)

#prediction = algo.predict(X_test_s['srch_id'],X_test_s['prop_id'],y_test_s,verbose=True)


## No transformation needed

# Create the parameter grid based on the results of random search
'''
param_grid = {
    'bootstrap': [True],
    'max_depth': [50, 75, 100],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [5, 10],
    'n_estimators': [50, 75, 100]
}
'''

#Create Classifier
#lg_clf = lightgbm.LGBMClassifier(objective='multiclass', num_class = 3)
#xgb_clf = XGBClassifier(objective='multi:softmax', num_class = 3)
#clf=RandomForestClassifier()

lg_ranker = lightgbm.LGBMRanker(objective='lambdarank') #, n_estimators=1000, learning_rate=0.001, num_leaves=50)


# Instantiate the grid search model
#grid_search = GridSearchCV(estimator = clf, param_grid = param_grid,
#                           cv = 3, n_jobs = 1, verbose = 2)
'''
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy
'''

# Fit the grid search to the data
#grid_search.fit(X_train_s, y_train_s)

#print(grid_search.best_params_)

#best_grid = grid_search.best_estimator_
#grid_accuracy = evaluate(best_grid, X_test_s, y_test_s)

#base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
#base_model.fit(X_train_s, y_train_s)
#base_accuracy = evaluate(base_model, X_test_s, y_test_s)

#print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


#Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train, y_train)

#clf.fit(X_train_s,y_train_s)
#xgb_clf.fit(X_train_s,y_train_s)
#lg_clf.fit(X_train_s,y_train_s)


qids_train = pd.DataFrame(X_train_s).groupby("srch_id")["srch_id"].count().to_numpy()
X_train_s = pd.DataFrame(X_train_s).drop(["srch_id"],axis=1)
qids_validation = X_test_s.groupby("srch_id")["srch_id"].count().to_numpy()
X_test_s = X_test_s.drop(["srch_id"], axis=1)

#print(qids_train)


group = qids_train
lg_ranker.fit(X_train_s,y_train_s,group=group,eval_set=[(X_test_s,y_test_s)],
              eval_group=[qids_validation],eval_metric='ndcg',eval_at=10,verbose=10)

#feature_imp = pd.Series(clf.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
#feature_imp = pd.Series(xgb_clf.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
#feature_imp = pd.Series(lg_clf.feature_importances_,index=list(X.columns)).sort_values(ascending=False)
feature_imp = pd.Series(lg_ranker.feature_importances_,index=list(X_train_s.columns)).sort_values(ascending=False)

print(feature_imp)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#y_pred_s = clf.predict(X_test_s)
#y_pred_s = xgb_clf.predict(X_test_s)
#y_pred_s = lg_clf.predict(X_test_s)
y_pred_s = lg_ranker.predict(X_test_s)

# Model Accuracy, how often is the classifier correct?
#cv_scores = cross_val_score(clf, X, y, cv=5)
#print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))

#print("Accuracy:",metrics.accuracy_score(y_test_s, y_pred_s))


'''
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    #disp = plot_confusion_matrix(clf, X_test_s, y_test_s,
    #                             display_labels=y_test_s.columns,
    #                             cmap=plt.cm.Blues,
    #                             normalize=normalize)
    #disp = plot_confusion_matrix(xgb_clf, X_test_s, y_test_s,
    #                             display_labels=y.unique().tolist(),
    #                             cmap=plt.cm.Blues,
    #                             normalize=normalize)
    disp = plot_confusion_matrix(lg_clf, X_test_s, y_test_s,
                                 display_labels=y.unique().tolist(),
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
plt.show()
'''

#pred = xgb_clf.predict_proba(X_test_s)
#pred = lg_clf.predict_proba(X_test_s)

#print(pred)

print(y_test_s,'\n', y_pred_s)

#new_pred = []

#for inner_list in pred:
#    new_pred.append(max(inner_list))

#print([new_pred])
#print([list(y_test_s)])

#new_pred = np.asarray([new_pred])
#y_test_s = np.asarray([list(y_test_s)])

#print("NDCG:",metrics.ndcg_score(y_test_s, new_pred))

print("NDCG:",metrics.ndcg_score(np.asarray([list(y_test_s)]),np.asarray([y_pred_s])))


'''
qids = train.groupby("srch_id")["srch_id"].count().to_numpy()
X = train.drop(["srch_id"],axis=1)

group = qids
lg_ranker.fit(X,y,group=group)
'''

## Get prediction
#y_pred = clf.predict(X_test)
y_pred = lg_ranker.predict(test)

## Save to df
test['y_pred'] = y_pred

## Print
print(y_pred)

## Create result
result = test[['srch_id','prop_id','y_pred']]

result = result.sort_values(['srch_id','y_pred'], ascending=(True,False))

## Create submission
submission = result[['srch_id', 'prop_id']]

#submission.to_csv('submission.csv',index=False)
submission.to_csv('submission_rank2.csv',index=False)
