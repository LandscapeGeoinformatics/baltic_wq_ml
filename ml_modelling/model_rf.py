import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from joblib import dump
import re


def get_group_key(col):
    """Find common covariate names with regex,
    like cropland, cropland_100, cropland_500"""
    m = re.match(r"^(.*?)(?:_(100|500))?_(mean|std)$", col)
    if m:
        return f"{m.group(1)}_{m.group(3)}"  # for numeric
    
    m = re.match(r"^(.*?)(?:_(100|500))?$", col) # for area percentage
    if m:
        return m.group(1)
    return col  

def retrieve_top_buffer(shap, X_train):
    """Returns the most important covariate of the three
    between the full-catchment, 100 m, and 500 m
    e.g., cropland, cropland_100m, cropland_500m"""

    # determine importance
    mean_abs_shap = np.abs(shap).mean(axis=0)
    
    shap_df = pd.Series(mean_abs_shap, index=X_train.columns)

    # find common covariate names (like cropland)
    group_keys = shap_df.index.to_series().apply(get_group_key)
    
    # find best of the three
    best_feature_per_group = shap_df.groupby(group_keys).idxmax()
    
    best_scores = shap_df.loc[best_feature_per_group].sort_values(ascending=False)
    
    return best_scores


## RF

def split_data(data, param, model_name, output_folder):
    
    Y = data['value']
    
    X = data.drop(['value'], axis=1)
    
    # Perform the random split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # save the splits for later reference
    X_train.to_csv(f'{output_folder}\\{model_name}_{param}_X_train.csv')
    X_test.to_csv(f'{output_folder}\\{model_name}_{param}_X_test.csv')
    Y_train.to_csv(f'{output_folder}\\{model_name}_{param}_Y_train.csv')
    Y_test.to_csv(f"{output_folder}\\{model_name}_{param}_Y_test.csv")
    
    # remove coordinates unless explicitly needed
    if model_name != 'coordinate':
        
        X_train = X_train.drop(['X', 'Y'], axis=1)
        X_test = X_test.drop(['X', 'Y'], axis=1)
    
    # remove the site_id - not a valid covariate
    X_train.drop('site_id', axis = 1, inplace = True)
    X_test.drop('site_id', axis = 1, inplace = True)
    
    y_train = np.ravel(Y_train)
    
    y_test = np.ravel(Y_test)
    
    # inspect if the correct covs are used
    print("Covariates used: ", ', '.join(X_train.columns))
    
    return X_train, X_test, y_train, y_test


def split_data_to_match(data, param, model_name, output_folder):
    # match the train-test split of the non-spatial models

    if param == 'tn':
    
        baseline_train = pd.read_csv(f'{output_folder}\\baseline_tn_X_train.csv')
    
    elif  param == 'tp':

        baseline_train = pd.read_csv(f'{output_folder}\\baseline_tp_X_train.csv')

    baseline_train = baseline_train[['site_id']]

    X_train = data[data['site_id'].isin(baseline_train['site_id'])]

    X_test = data[~data['site_id'].isin(baseline_train['site_id'])]

    Y_train = X_train['value']
    
    Y_test = X_test['value']

    X_train = X_train.drop(['value'], axis=1)

    X_test = X_test.drop(['value'], axis=1)

    # save the splits for later reference
    X_train.to_csv(f'{output_folder}\\{model_name}_{param}_X_train.csv')
    X_test.to_csv(f'{output_folder}\\{model_name}_{param}_X_test.csv')
    Y_train.to_csv(f'{output_folder}\\{model_name}_{param}_Y_train.csv')
    Y_test.to_csv(f"{output_folder}\\{model_name}_{param}_Y_test.csv")
    
    # remove coordinates unless explicitly needed
    if model_name != 'coordinate':
        
        X_train = X_train.drop(['X', 'Y'], axis=1)
        X_test = X_test.drop(['X', 'Y'], axis=1)
    
    # remove the site_id - not a valid covariate
    X_train = X_train.drop('site_id', axis = 1)
    X_test = X_test.drop('site_id', axis = 1)
    
    y_train = np.ravel(Y_train)
    
    y_test = np.ravel(Y_test)
    
    # inspect if the correct covs are used
    print("Covariates used: ", ', '.join(X_train.columns))
    
    return X_train, X_test, y_train, y_test


def tune_hyperparameters(X_train, y_train):
    
    param_grid = { 
    'n_estimators': list(np.arange(50, 201, 10).astype(int)),
        
    'max_depth': list(np.arange(5, 21, 5, dtype=int)),
        
   } 
    
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    
    grid_search = GridSearchCV(RandomForestRegressor(random_state=0), 
                           param_grid=param_grid, cv=cv, verbose=3, n_jobs=1) 
    
    grid_search.fit(X_train, y_train) 
    
    # extract the best hyperparameters
    hyperparameters = grid_search.best_params_
    
    hyperparameters['oob_score'] = True

    return hyperparameters


def train_model(hyperparameters, X_train, y_train):
    
    print("Model trained using the following hyperparameters: ", hyperparameters)

    rf = RandomForestRegressor(random_state=0).set_params(**hyperparameters)
    
    rfReg = rf.fit(X_train, y_train)

    

    return rfReg


def calc_score(rfReg, X_train, X_test, y_train, y_test):
    
    train = rfReg.predict(X_train)
    
    test = rfReg.predict(X_test)
    
    print("Rsq:", round(r2_score(y_train, train), 2), round(r2_score(y_test, test), 2))
    
    print("OOB:", round(rfReg.oob_score_, 2))
    
    print("MAE:", round(mean_absolute_error(y_train, train), 2), round(mean_absolute_error(y_test, test), 2))
    
    return train, test


def save_predictions(train, test, y_train, y_test, model_name, param, output_folder):
    """Reconstruct a file with covariates, the target, the residual, and predicted values.
    Name after the model_name""" 
    
    X_train = pd.read_csv(f'{output_folder}\\{model_name}_{param}_X_train.csv')
    
    X_test = pd.read_csv(f'{output_folder}\\{model_name}_{param}_X_test.csv')
    
    X_train['pred'] = train

    X_test['pred']  = test
    
    X_train['obs'] = y_train
    
    X_test['obs'] = y_test
    
    predictions = pd.concat([X_train, X_test])
    
    predictions['residual'] = predictions['pred'] - predictions['obs'] 

    predictions['perc_res'] = (predictions['residual'] / predictions['obs']) * 100
    
    predictions.to_csv(f'{output_folder}\\predictions_{model_name}_{param}.csv')


def model_wq(data, param, model_name, output_folder):
    
    print(f"RF model for {param}")
    
    print(f'Splitting data for {param}')

    # Train-test split: from scratch for non-spatial models, to match for spatial models
    if model_name == 'baseline' or model_name == 'top5':
        
        X_train, X_test, y_train, y_test = split_data(data, param, model_name, output_folder)

    else:
        
        X_train, X_test, y_train, y_test = split_data_to_match(data, param, model_name, output_folder)

    print(f'Tuning hyperparameters for {param}')
    hyperparameters = tune_hyperparameters(X_train, y_train)
    
    print(f'Training the best model for {param}')
    rf_model = train_model(hyperparameters, X_train, y_train, )
    
    print(f'Saving the model for {param}')
    dump(rf_model, f'{output_folder}/{param}_{model_name}_model.joblib')

    print(f'Calculating the score for {param}')
    preds_train, preds_test = calc_score(rf_model, X_train, X_test, y_train, y_test)
    
    print(f'Exporting the predictions')
    save_predictions(preds_train, preds_test, y_train, y_test, model_name, param, output_folder)

    return rf_model, X_train
