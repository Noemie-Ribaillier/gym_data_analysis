########################################################################################################
#####                                                                                              #####
#####                         DATA PROJECT ABOUT CALORIES BURNT AT THE GYM                         #####
#####                                      Noemie RIBAILLIER                                       #####
#####                                    Created on: 2024-10-22                                    #####
#####                                    Updated on: 2025-03-09                                    #####
#####                                                                                              #####
########################################################################################################

########################################################################################################
#####                                      LOAD THE LIBRARIES                                      #####
########################################################################################################

# Clean the whole environment
globals().clear()

# Load the libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as metrics
from xgboost import XGBRegressor
import pickle
import streamlit as st

# Set up the right directory
import os 
os.chdir('C:/Users/Admin/Documents/Python Projects/gym_data_analysis')
from general_functions import *


########################################################################################################
#####                        LOAD THE DATA AND DO A QUICK DATA PREPROCESSING                       #####
########################################################################################################

# Load the data and check the dimensions, head, column names and main info/statistics
gym_data = pd.read_csv('gym_data.csv')
gym_data.shape
gym_data.head
gym_data.columns
gym_data.info()
gym_data.describe(include='all')

# Lower case the columns
gym_data.columns = gym_data.columns.str.lower()

# Rename the columns
gym_data = gym_data.rename(columns={"weight (kg)": "weight", "height (m)": "height",
                         'session_duration (hours)':'session_duration',
                         'water_intake (liters)':'water_intake',
                         'workout_frequency (days/week)':'workout_frequency',
                         "calories_burned": "calories_burnt"})

# Handle categorical variables (to optimize memory and performance)  
# Check the type of each variable
gym_data.dtypes 
categorical_variable = ['gender','workout_type','workout_frequency','experience_level']
gym_data[categorical_variable] = gym_data[categorical_variable].astype('category')

# Double check the column names and column types
gym_data.dtypes


########################################################################################################
#####                                     UNIVARIATE ANALYSIS                                      #####
########################################################################################################

# GENDER
categorical_univariate_analysis(gym_data,'gender')

# AGE
numerical_univariate_analysis(gym_data,['age'],'Age')

# BPM
numerical_univariate_analysis(gym_data,['resting_bpm','avg_bpm','max_bpm'],'BPM')

# WEIGHT
numerical_univariate_analysis(gym_data,['weight'],'Weight')

# HEIGHT
numerical_univariate_analysis(gym_data,['height'],'Height')

# SESSION_DURATION
numerical_univariate_analysis(gym_data,['session_duration'],'Session duration')

# CALORIES_BURNT
numerical_univariate_analysis(gym_data,['calories_burnt'],'Calories burnt')

# WORKOUT_TYPE
categorical_univariate_analysis(gym_data,'workout_type')

# FAT_PERCENTAGE
numerical_univariate_analysis(gym_data,['fat_percentage'],'Fat percentage')

# WATER_INTAKE
numerical_univariate_analysis(gym_data,['water_intake'],'Water intake')

# WORKOUT_FREQUENCY
categorical_univariate_analysis(gym_data,'workout_frequency')

# EXPERIENCE_LEVEL
categorical_univariate_analysis(gym_data,'experience_level')

# BMI
numerical_univariate_analysis(gym_data,['bmi'],'BMI')
# Check that bmi is well the weight divided by the square of the size
gym_data['computed_bmi'] = round(gym_data['weight']/(gym_data['height'])**2,2)
gym_data.loc[gym_data['computed_bmi']!=gym_data['bmi'],['bmi','weight','height','computed_bmi']]
# Get the proportion per class of BMI
np.round(gym_data.loc[(gym_data['bmi']<18.5)].shape[0]*100/gym_data.shape[0],2)
np.round(gym_data.loc[(gym_data['bmi']>=18.5) & (gym_data['bmi']<25)].shape[0]*100/gym_data.shape[0],2)
np.round(gym_data.loc[(gym_data['bmi']>=25) & (gym_data['bmi']<30)].shape[0]*100/gym_data.shape[0],2)
np.round(gym_data.loc[(gym_data['bmi']>=30)].shape[0]*100/gym_data.shape[0],2)


########################################################################################################
#####                                      BIVARIATE ANALYSIS                                      #####
########################################################################################################

# AGE
numerical_numerical_bivariate_analysis(gym_data,'age','calories_burnt')

# WEIGHT
numerical_numerical_bivariate_analysis(gym_data,'weight','calories_burnt')

# HEIGHT
numerical_numerical_bivariate_analysis(gym_data,'height','calories_burnt')

# MAX_BPM
numerical_numerical_bivariate_analysis(gym_data,'max_bpm','calories_burnt')

# AVG_BPM
numerical_numerical_bivariate_analysis(gym_data,'avg_bpm','calories_burnt')

# RESTING_BPM
numerical_numerical_bivariate_analysis(gym_data,'resting_bpm','calories_burnt')

# SESSION_DURATION
numerical_numerical_bivariate_analysis(gym_data,'session_duration','calories_burnt')

# FAT_PERCENTAGE
numerical_numerical_bivariate_analysis(gym_data,'fat_percentage','calories_burnt')

# WATER_INTAKE
numerical_numerical_bivariate_analysis(gym_data,'water_intake','calories_burnt')

# BMI
numerical_numerical_bivariate_analysis(gym_data,'bmi','calories_burnt')

# GENDER
numerical_categorical_bivariate_analysis(gym_data,'gender','calories_burnt')

# WORKOUT_TYPE
numerical_categorical_bivariate_analysis(gym_data,'workout_type','calories_burnt')

# EXPERIENCE_LEVEL
numerical_categorical_bivariate_analysis(gym_data,'experience_level','calories_burnt')

# WORKOUT_FREQUENCY
numerical_categorical_bivariate_analysis(gym_data,'workout_frequency','calories_burnt')


########################################################################################################
#####                                           ENCODING                                           #####
########################################################################################################

# Encode the nominal categorical variables
gym_data_encoded = gym_data.copy()

# Encode gender (transform from Female/Male to 0/1)
gender_encoder = LabelEncoder()
gym_data_encoded['gender'] = gender_encoder.fit_transform(gym_data_encoded['gender'])

# Encode workout_type (transform from 'Cardio' to [1 0 0], from 'HIIT' to [0 1 0], from 'Strength' to [0 0 1] and from 'Yoga' to [0 0 0])
# Step 1: Convert to a 2D numpy array (OneHotEncoder expects 2D input)
workout_type_array = np.array(gym_data_encoded['workout_type']).reshape(-1, 1)
# Step 2: Initialize OneHotEncoder
workout_type_encoder = OneHotEncoder(sparse_output=False)
# Step 3: Fit and transform the data
workout_type_encoded = workout_type_encoder.fit_transform(workout_type_array)
# Step 4: Convert to a DataFrame for better readability (optional)
categories_lower = [category.lower() for category in workout_type_encoder.categories_[0]]
workout_type_encoded_df = pd.DataFrame(workout_type_encoded, columns=categories_lower)
# Step 5: Join with the gym data and drop yoga variable (to remove multicolinearity) and workout_type (because we encoded it)
gym_data_encoded = gym_data_encoded.join(workout_type_encoded_df)
gym_data_encoded = gym_data_encoded.drop(['yoga','workout_type'],axis=1)

# Transform the ordinal categorical variables to numerical
gym_data_encoded[['experience_level', 'workout_frequency', 'cardio', 'hiit', 'strength']] = gym_data_encoded[['experience_level', 'workout_frequency', 'cardio', 'hiit', 'strength']].astype('int')

# Drop bmi variable since it's computed from height and weight so multicolinearity
gym_data_encoded = gym_data_encoded.drop('bmi',axis=1)


# Check the dimensions and encoding
gym_data_encoded.shape
gym_data_encoded.columns
gym_data_encoded.dtypes
gym_data_encoded[['gender','cardio','hiit','strength']].head()
gym_data[['gender','workout_type']].head()


########################################################################################################
#####                            SPLIT DATASET INTO TRAIN AND TEST SET                             #####
########################################################################################################

# Split data into features (X) and label (Y)
X = gym_data_encoded.drop('calories_burnt', axis = 1)
y = gym_data_encoded.calories_burnt
print(X.head())
print(y.head())

# Split dataset into train and test (random_state to get reproducible results)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Print the shape of the datasets we just created
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Normalize the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set up the kfold CV, that we will use for all models
kf = KFold(n_splits=5, shuffle=True, random_state = 21)

# Create lists to gather all statistics for all models
all_mae=[]
all_mse=[]
all_rmse=[]
all_r2=[]
all_models=[]


########################################################################################################
#####                                DECISION TREES REGRESSOR MODEL                                #####
########################################################################################################

# Initialize the Decision Tree Regressor
decision_tree_regressor = DecisionTreeRegressor(random_state=21)

# Hyperparameters grid
decision_tree_param_grid = {
    # Measure the quality of a split
    'criterion': ['squared_error'],
    # Maximum depth of the tree
    'max_depth': [3, 5, 7],
    # Minimum number of samples required to split an internal node
    'min_samples_split': [3, 5, 10],
    # Minimum number of samples required to be at a leaf node
    'min_samples_leaf': [2, 4, 6],
    # Number of features to consider for the best split
    'max_features': [None, 'sqrt', 12],
    # Strategy used to split at each node
    'splitter': ['best', 'random'],
    # Maximum number of leaf nodes
    'max_leaf_nodes': [None, 10, 20, 30]
}

# Create Decision Trees regressor model (get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models)
all_mae, all_mse, all_rmse, all_r2, all_models = regression_model(
    decision_tree_regressor, decision_tree_param_grid, kf, X_train, y_train, X_test, y_test, 
    all_mae, all_mse, all_rmse, all_r2, all_models)


########################################################################################################
#####                                   K-NEAREST NEIGHBOR MODEL                                   #####
########################################################################################################

# Initialize the K-Nearest Neighbor Regressor
knn_regressor = KNeighborsRegressor()

# Hyperparameters grid
knn_param_grid = {
    # number of neighbors to use for prediction
    'n_neighbors': [3, 5, 10, 20],
    # weight function used in prediction
    'weights': ['uniform', 'distance'],
    # algorithm to compute the nearest neighbors
    'algorithm': ['auto'],
    # distance metric to use
    'metric': ['euclidean']
}

# Create KNN regressor model (get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models)
all_mae, all_mse, all_rmse, all_r2, all_models = regression_model(
    knn_regressor, knn_param_grid, kf, X_train, y_train, X_test, y_test, 
    all_mae, all_mse, all_rmse, all_r2, all_models)


########################################################################################################
#####                                RANDOM FOREST REGRESSOR MODEL                                 #####
########################################################################################################

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=21)

# Hyperparameters grid
rf_param_grid = {
    # Number of trees in the forest
    'n_estimators': [100, 200, 300, 400, 500],
    # Maximum depth of each tree
    'max_depth': [3, 5, 7],
    # Minimum number of samples required to split a node
    'min_samples_split': [3, 5, 10],
    # Minimum number of samples required at a leaf node
    'min_samples_leaf': [2, 4, 6],
    # Number of features to consider when looking for the best split
    'max_features': [12, 'sqrt'],
    # Whether bootstrap samples are used when building trees
    'bootstrap': [True],
    # Whether to reuse the solution of the previous call to fit and add more estimators
    'warm_start': [False]
}

# Create Random Forest regressor model (get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models)
all_mae, all_mse, all_rmse, all_r2, all_models = regression_model(
    rf_regressor, rf_param_grid, kf, X_train, y_train, X_test, y_test, 
    all_mae, all_mse, all_rmse, all_r2, all_models)


########################################################################################################
#####                              GRADIENT BOOSTING REGRESSOR MODEL                               #####
########################################################################################################

# Initialize the GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor(random_state = 21)

# Hyperparameters grid
gb_param_grid = {
    # Number of boosting stages (trees)
    'n_estimators': [100, 200, 300, 400, 500],
    # Learning rate
    'learning_rate': [0.01, 0.05, 0.1],
    # Maximum depth of individual trees
    'max_depth': [3, 5, 7],
    # Fraction of samples used for fitting each tree
    'subsample': [0.8, 1.0],
    # Minimum samples required to split a node
    'min_samples_split': [3, 5, 10],
    # Minimum samples required at a leaf node
    'min_samples_leaf': [2, 4, 6],
    # Number of features to consider when splitting a node
    'max_features': [12, 'sqrt']
}

# Create Gradient Boosting regressor model (get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models)
all_mae, all_mse, all_rmse, all_r2, all_models = regression_model(
    gb_regressor, gb_param_grid, kf, X_train, y_train, X_test, y_test, 
    all_mae, all_mse, all_rmse, all_r2, all_models)


########################################################################################################
#####                                   XGBOOST REGRESSOR MODEL                                   #####
########################################################################################################

# Initialize the XGBoost Regressor
xgboost_regressor = XGBRegressor(enable_categorical=True, random_state=21)

# Hyperparameters grid
xgboost_param_grid = {
    # Number of boosting rounds (trees)
    'n_estimators': [100, 200, 300, 400, 500],
    # Step size for each boosting round
    'learning_rate': [0.01, 0.05, 0.1],
    # Maximum depth of individual trees
    'max_depth': [3, 5, 7],
    # Minimum sum of instance weights in a child
    'min_child_weight': [1, 5, 10],
    # Fraction of samples used for fitting each tree
    'subsample': [0.8, 1.0], #0.9, 
     # Fraction of features used for each tree
    'colsample_bytree': [0.8, 1.0], #0.9, 
    # Minimum loss reduction required to make a further partition
    'gamma': [0, 0.1, 0.2],
    # L1 regularization term on weights
    'reg_alpha': [0, 0.01, 0.1],
    # # L2 regularization term on weights
    'reg_lambda': [1, 1.5, 2],
    # Loss function for regression
    'objective': ['reg:squarederror'],
    # Type of boosting model (tree-based, linear, or DART)
    'booster': ['gbtree']
}

# Create XGBoost regressor model (get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models)
all_mae, all_mse, all_rmse, all_r2, all_models = regression_model(
    xgboost_regressor, xgboost_param_grid, kf, X_train, y_train, X_test, y_test, 
    all_mae, all_mse, all_rmse, all_r2, all_models)


########################################################################################################
#####                            SUPPORT VECTOR MACHINE REGRESSOR MODEL                            #####
########################################################################################################

# Initialize the Support Vector Machine Regressor
svr_regressor = SVR()

# Hyperparameters grid
svr_param_grid = {
    # The type of kernel to use
    'kernel': ['linear', 'poly', 'rbf'],
    # Regularization parameter (penalty for errors)
    'C': [0.1, 1, 10, 100],  
    # Epsilon parameter, controls margin of tolerance
    'epsilon': [0.01, 0.1, 1, 10],
    # Degree of the polynomial kernel function (only relevant for 'poly')
    'degree': [2, 3, 4],
    # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'gamma': ['scale', 'auto', 0.001, 0.01],
    # Tolerance for stopping criteria
    'tol': [1e-3, 1e-4, 1e-5],
    # Size of the kernel cache (MB)
    'cache_size': [200, 500],
    # Whether to use the shrinking heuristic
    'shrinking': [True, False]
}

# Create SVR regressor model (get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models)
all_mae, all_mse, all_rmse, all_r2, all_models = regression_model(
    svr_regressor, svr_param_grid, kf, X_train, y_train, X_test, y_test, 
    all_mae, all_mse, all_rmse, all_r2, all_models)


########################################################################################################
#####                         COMPARE ALL THE MODELS AND SAVE THE BEST ONE                         #####
########################################################################################################

# Pick the lowest RMSE
index_min_rmse = all_rmse.index(min(all_rmse))
print('RMSE of the best model is (min RMSE among all trained models) is: '+str(np.round(all_rmse[index_min_rmse],2)))
print('MAE of the best model is (min MAE among all trained models) is: '+str(np.round(all_mae[index_min_rmse],2)))

# Check the R2 of the best model
print('R2 of the best model is: '+str(np.round(all_r2[index_min_rmse],2)*100)+'%')

# Get the best model (with the lowest RMSE)
best_model = all_models[index_min_rmse]
model_name = best_model.__class__.__name__
print(f"The best model (among all trained models) is: {model_name}")
print("The best model (among all trained models) with its tuned parameters is: "+str(best_model))

# Save the model (and all the encoders)
saved_model = {"model": best_model,  "gender_encoder": gender_encoder, "workout_type_encoder": workout_type_encoder}
with open('C:/Users/Admin/Documents/Python Projects/gym_data_analysis/best_model.pkl', 'wb') as file:
    pickle.dump(saved_model,file)

