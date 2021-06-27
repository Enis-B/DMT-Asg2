import pandas as pd
import numpy as np
import shap
from catboost import CatBoost, Pool
shap.initjs()
from copy import deepcopy


pd.options.display.float_format = '{:,.3f}'.format


def get_target(row):
    """
    0=not clicked at all, 1=clicked but not booked, 5=booked
    """
    if row.booking_bool>0:
        return 1
    if row.click_bool>0 :
        return 0.2
    return 0


def featurize_df(df:pd.DataFrame) ->pd.DataFrame:
    """
    Extract more features
    """
    df["weekday"] = df["date_time"].dt.isocalendar().week
    df["week_of_year"] = df["date_time"].dt.isocalendar().week

    df["hour"] = df["date_time"].dt.hour
    df["minute"] = df["date_time"].dt.minute
    ## total time elapsed - allows model to learn continous trend over time to a degree
    df["time_epoch"] = df["date_time"].astype('int64')//1e9
    ## if we were looking at fraud: df["seconds"] = df.timestamp.dt.second
    df["early_night"] = ((df["hour"]>19) | (df["hour"]<3)) # no added value from feature

    df["nans_count"] = df.isna().sum(axis=1)

    ## we won't make any time series features for now
    ## We could add time series features per property/hotel. We'd need to check for unaries, and to add a shift/offset dependant on forecast horizon

    return df

df = pd.read_csv('training_set_VU_DM.csv')
print(df.shape)

df["date_time"] = pd.to_datetime(df["date_time"],infer_datetime_format=True)
df["target"] = df.apply(get_target,axis=1)
# featurization must be after leaky cols are dropped, otherwise the nan feature will bea leak!
df.describe()
df.nunique()

df["date_time"].describe(datetime_is_numeric=True)
print(df)


df_test = pd.read_csv('test_set_VU_DM.csv')
print(df_test.shape)
cols = df_test.columns.drop(['date_time'])
float_cols = df_test.columns[df_test.dtypes.eq('float')]# float_cols = df.columns.drop(['date_time'])
for c in float_cols:
    df_test[c] = pd.to_numeric(df_test[c], errors="ignore",downcast="integer")
df_test["date_time"] = pd.to_datetime(df_test["date_time"],infer_datetime_format=True)

print(df_test)


df.drop_duplicates(['click_bool','booking_bool','random_bool'])


drop_cols = []

## we see many columns are unary - drop them, barring feature engineering
drop_unary_cols = [c for c
                   in list(df)
                   if df[c].nunique(dropna=False) <= 1]
target_cols = ["gross_bookings_usd","click_bool","booking_bool"] # leaky column, and original target columns
drop_cols.extend(drop_unary_cols)
drop_cols.extend(target_cols)

### we'll need to remove datetime from the model, but it may be useful for train/test split before that
# drop_cols.append("date_time")

df = df.drop(columns=drop_cols,errors="ignore")
df_test = df_test.drop(columns=drop_cols,errors="ignore")
print(df.shape)
print(df)


df = featurize_df(df)
df_test = featurize_df(df_test)


## sort by high rank, regardless of booked or not (for easy comp)
df.drop(['comp3_rate',
         'comp3_inv', 'comp3_rate_percent_diff', 'comp4_inv', 'comp5_rate',
         'comp5_inv', 'comp5_rate_percent_diff', 'comp8_rate', 'comp8_inv',
         'comp8_rate_percent_diff'],axis=1).groupby(df["target"]>0).mean()

cutoff_id = df["srch_id"].quantile(0.94) # 90/10 split
X_train = df.loc[df.srch_id< cutoff_id].drop(["target"],axis=1)
X_eval = df.loc[df.srch_id>= cutoff_id].drop(["target"],axis=1)
y_train = df.loc[df.srch_id< cutoff_id]["target"]
y_eval = df.loc[df.srch_id>= cutoff_id]["target"]




print("mean relevancy train",round(y_train.mean(),4))
print("mean relevancy eval",round(y_eval.mean(),4))
print(y_eval.value_counts()) # check we have all 3 "labels" in subset


df["target"].value_counts()


categorical_cols = ['prop_id',"srch_destination_id", "weekday"] # ,"week_of_year"


print(df.tail())


## check for feature/column leaks
set(X_train.columns).symmetric_difference(set(df_test.columns))


train_pool = Pool(data=X_train,
                  label = y_train,
                  cat_features=categorical_cols,
                  group_id=X_train["srch_id"],
                  )

eval_pool = Pool(data=X_eval,
                 label = y_eval,
                 cat_features=categorical_cols,
                 group_id=X_eval["srch_id"]
                 )



default_parameters  = {
'iterations': 2000,
'custom_metric': ['NDCG', "AUC:type=Ranking"], # , 'AverageGain:top=3'# 'QueryRMSE', "YetiLoss" (use with hints)
'verbose': False,
'random_seed': 42,
     "task_type":"GPU",
"has_time":True,
"metric_period":4,
"save_snapshot":False,
"use_best_model":True, # requires eval set to be set
}

parameters = {}

def fit_model(loss_function, additional_params=None, train_pool=train_pool, test_pool=eval_pool):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    print("best results (train on train):")
    print(model.get_best_score()["learn"])
    print("best results (on validation set):")
    print(model.get_best_score()["validation"])

    print("(Default) Feature importance (on train pool)")
    print(model.get_feature_importance(data=train_pool,prettified=True).head(15))

    try:
        print("SHAP features importance, on all data:")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.concat([X_train,X_eval]),
                                            y=pd.concat([y_train,y_eval]))

        # # summarize the effects of all the features
        shap.summary_plot(shap_values, pd.concat([X_train,X_eval]))
    finally:
        return model

