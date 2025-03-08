########################################################################################################
#####                                                                                              #####
#####                        DATA PROJECT ABOUT CALORIES BURNED AT THE GYM                         #####
#####                                      Noemie RIBAILLIER                                       #####
#####                                    Created on: 2024-10-22                                    #####
#####                                    Updated on: 2025-03-06                                    #####
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
                         'workout_frequency (days/week)':'workout_frequency'})

# Handle categorical variables (to optimize memory and performance)  
# Check the type of each variable
gym_data.dtypes 
categoric_variable = ['gender','workout_type','workout_frequency','experience_level']
gym_data[categoric_variable] = gym_data[categoric_variable].astype('category')


# Double check the column names and column types
gym_data.dtypes


########################################################################################################
#####                                     UNIVARIATE ANALYSIS                                      #####
########################################################################################################

# GENDER
categoric_univariate_analysis(gym_data,'gender')

# AGE
numeric_univariate_analysis(gym_data,['age'],'Age')

# BPM
numeric_univariate_analysis(gym_data,['resting_bpm','avg_bpm','max_bpm'],'BPM')

# WEIGHT
numeric_univariate_analysis(gym_data,['weight'],'Weight')

# HEIGHT
numeric_univariate_analysis(gym_data,['height'],'Height')

# SESSION_DURATION
numeric_univariate_analysis(gym_data,['session_duration'],'Session duration')

# CALORIES_BURNED
numeric_univariate_analysis(gym_data,['calories_burned'],'Calories burned')

# WORKOUT_TYPE
categoric_univariate_analysis(gym_data,'workout_type')

# FAT_PERCENTAGE
numeric_univariate_analysis(gym_data,['fat_percentage'],'Fat percentage')

# WATER_INTAKE
numeric_univariate_analysis(gym_data,['water_intake'],'Water intake')

# WORKOUT_FREQUENCY
categoric_univariate_analysis(gym_data,'workout_frequency')

# EXPERIENCE_LEVEL
categoric_univariate_analysis(gym_data,'experience_level')

# BMI
numeric_univariate_analysis(gym_data,['bmi'],'BMI')
# Check that bmi is well the weight divided by the square of the size
gym_data['computed_bmi'] = round(gym_data['weight']/(gym_data['height'])**2,2)
gym_data.loc[gym_data['computed_bmi']!=gym_data['bmi'],['bmi','weight','height','computed_bmi']]


########################################################################################################
#####                                      BIVARIATE ANALYSIS                                      #####
########################################################################################################

# AGE
numeric_bivariate_analysis(gym_data,'age','calories_burned')

# WEIGHT
numeric_bivariate_analysis(gym_data,'weight','calories_burned')

# HEIGHT
numeric_bivariate_analysis(gym_data,'height','calories_burned')

# MAX_BPM
numeric_bivariate_analysis(gym_data,'max_bpm','calories_burned')

# AVG_BPM
numeric_bivariate_analysis(gym_data,'avg_bpm','calories_burned')

# RESTING_BPM
numeric_bivariate_analysis(gym_data,'resting_bpm','calories_burned')

# SESSION_DURATION
numeric_bivariate_analysis(gym_data,'session_duration','calories_burned')

# FAT_PERCENTAGE
numeric_bivariate_analysis(gym_data,'fat_percentage','calories_burned')

# WATER_INTAKE
numeric_bivariate_analysis(gym_data,'water_intake','calories_burned')

# BMI
numeric_bivariate_analysis(gym_data,'bmi','calories_burned')

# GENDER
categoric_bivariate_analysis(gym_data,'gender','calories_burned')

# WORKOUT_TYPE
categoric_bivariate_analysis(gym_data,'workout_type','calories_burned')

# EXPERIENCE_LEVEL
categoric_bivariate_analysis(gym_data,'experience_level','calories_burned')

# WORKOUT_FREQUENCY
categoric_bivariate_analysis(gym_data,'workout_frequency','calories_burned')

# 








########################################################################################################
#####                                   2 CATEGORICAL VARIABLES                                    #####
########################################################################################################

# This is a categoric to categoric relationship, methods:
# chi-squared test: determine association between categorical variabes
# flat/stacked bar chart

# Compare the relationship between gender and workout type

# Plot
subset_gender_workouttype = gym_data[['gender','workout_type','bmi']]
genders = sorted(subset_gender_workouttype['gender'].unique().tolist())
workout_types = sorted(subset_gender_workouttype['workout_type'].unique().tolist())
subset_gender_workouttype_pt = np.array(pd.pivot_table(subset_gender_workouttype, index=[ 'gender'],columns='workout_type',aggfunc='count'))
subset_gender_workouttype_summary = np.round(subset_gender_workouttype_pt*100/subset_gender_workouttype_pt.sum(axis=1)[:,None],1)
print(subset_gender_workouttype_summary)
plt.figure(figsize=(10, 6))
plt.bar(genders, subset_gender_workouttype_summary[:, 0], label=workout_types[0])
for i in range(1, subset_gender_workouttype_summary.shape[1]):
    plt.bar(genders, subset_gender_workouttype_summary[:, i], bottom=subset_gender_workouttype_summary[:, :i].sum(axis=1), label=workout_types[i])
plt.xlabel('Genders')
plt.ylabel('Pct')
plt.title('Difference of the workout type per gender')
plt.legend(title='Workout type',loc='upper right')
plt.tight_layout()
plt.xticks(genders,genders)
plt.show()


# TEST
# contingency table (pivot table)
contingency_table = pd.crosstab(gym_data['gender'],gym_data['workout_type'])
# Formulating hypothesis
# H0: there is no relationship between gender and workout_type
# H1: there is a significant association between the 2 variables
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(chi2_stat, p_value, dof, expected)
# Interpreting results
# pvalue=0.7>0.05, meaning we don't reject H0


# Compare the relationship between experience level and workout frequency

# PLot
subset_explevel_workoutfreq = gym_data[['experience_level','workout_frequency','bmi']]
experience_levels = sorted(subset_explevel_workoutfreq['experience_level'].unique().tolist())
workout_frequencies = sorted(subset_explevel_workoutfreq['workout_frequency'].unique().tolist())
subset_explevel_workoutfreq_pt = np.array(pd.pivot_table(subset_explevel_workoutfreq, index=[ 'experience_level'],columns='workout_frequency',aggfunc='count'))
subset_explevel_workoutfreq_summary = np.round(subset_explevel_workoutfreq_pt*100/subset_explevel_workoutfreq_pt.sum(axis=1)[:,None],1)
print(subset_explevel_workoutfreq_summary)
plt.figure(figsize=(10, 6))
plt.bar(experience_levels, subset_explevel_workoutfreq_summary[:, 0], label=workout_frequencies[0])
for i in range(1, subset_explevel_workoutfreq_summary.shape[1]):
    plt.bar(experience_levels, subset_explevel_workoutfreq_summary[:, i], bottom=subset_explevel_workoutfreq_summary[:, :i].sum(axis=1), label=workout_frequencies[i])
plt.xlabel('Experience levels')
plt.ylabel('Pct')
plt.title('Difference of the workout frequency per experience level')
plt.legend(title='Workout frequencies',loc='upper right')
plt.tight_layout()
plt.xticks(experience_levels,experience_levels)
plt.show()
# We see that low experienced people only practice the less per week while high experienced people practice the most per week

# TEST
contingency_table = pd.crosstab(gym_data['workout_frequency'],gym_data['experience_level'])
# Formulating hypothesis
# H0: there is no relationship between experience level and workout_frequency
# H1: there is a significant association between the 2 variables
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(chi2_stat, p_value, dof, expected)
# Interpreting results
# pvalue<2.2e-16<0.05, meaning we reject H0, the 2 variables are not independent
observed = contingency_table.values
residuals = (observed - expected)/np.sqrt(expected)
print(residuals)
# residuals of exp=1 and freq=2 are positive and pretty large, meaning that the observed count is higher than the expected count
# residuals of exp=3 and freq=5 are positive and very large, meaning that the observed count is way higher than the expected count
# negative residuals mean that the observed count < expected count
# contribution diagram:
contributions = (observed-expected)**2/expected
total_chi_square = chi2_stat
percentage_contributions = 100 * contributions / total_chi_square
print(np.round(percentage_contributions, 2))

# Plot the contributions
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    percentage_contributions,
    annot=True,          # Display numbers
    fmt=".2f",          # Format of the numbers (adjust as needed)
    cmap="coolwarm",     # Color map (adjust as needed)
    cbar_kws={"label": "Percentage Contribution"}, # Color bar label
    linewidths=0.5      # Space between cells
)
# Set the title
plt.title("Percentage contribution to Chi-Square statistic")
# Display the heatmap
plt.tight_layout()
plt.show()
# we just what we just noticed: what's contributing the most to the chi2 stat is the exp=3 and freq=5


########################################################################################################
#####                                           ENCODING                                           #####
########################################################################################################

# Encode the categorical nominal variables
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


# Check the dimensions and encoding
gym_data_encoded.shape
gym_data_encoded.columns
gym_data_encoded[['gender','cardio','hiit','strength']].head()
gym_data[['gender','workout_type']].head()


########################################################################################################
#####                            SPLIT DATASET INTO TRAIN AND TEST SET                             #####
########################################################################################################

# Split data into features (X) and label (Y)
X = gym_data_encoded.drop('calories_burned', axis = 1)
y = gym_data_encoded.calories_burned
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
    'max_depth': [5, 10, 20], #try with 3,5,7
    # Minimum number of samples required to split an internal node
    'min_samples_split': [3, 5, 10],
    # Minimum number of samples required to be at a leaf node
    'min_samples_leaf': [2, 4, 6],
    # Number of features to consider for the best split
    'max_features': [None, 'sqrt', 'log2'],
    # Strategy used to split at each node
    'splitter': ['best', 'random'],
    # Maximum number of leaf nodes
    'max_leaf_nodes': [None, 10, 20, 30],
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


# R2: 76.45%
# MAE: 96.39
# MSE: 14832.08
# RMSE: 121.79


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
    'max_depth': [5, 10, 20],               #try with 3,5,7
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
    'n_estimators': [100, 200, 300],
    # Step size for each boosting round
    'learning_rate': [0.01, 0.05, 0.1],
    # Maximum depth of individual trees
    'max_depth': [3, 5, 7],
    # Minimum sum of instance weights in a child
    'min_child_weight': [1, 5, 10],
    # Fraction of samples used for fitting each tree
    'subsample': [0.8, 0.9, 1.0],
     # Fraction of features used for each tree
    'colsample_bytree': [0.8, 0.9, 1.0],
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








# # Initialize GridSearchCV for hyperparameter tuning with CV
# # n_jobs=-1: using all processors
# # verbose=1: doesn't print anything (otherwise it prints a line for each combination of parameters tested and each CV folder)
# xgboost_grid_search = GridSearchCV(estimator = xgboost_regressor, param_grid = xgboost_param_grid, cv = kf, 
#                               scoring='neg_root_mean_squared_error', n_jobs = -1, verbose = 1)

# # Fit the model with GridSearchCV 
# xgboost_grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# print("Best Hyperparameters:", xgboost_grid_search.best_params_)

# # Get the best model from grid search
# xgboost_best_model = xgboost_grid_search.best_estimator_

# # Train the model
# xgboost_best_model.fit(X_train, y_train)

# # Predict on the test set
# xgboost_y_pred = xgboost_best_model.predict(X_test)

# # Evaluate the model on the test set
# xgboost_mae, xgboost_mse, xgboost_rmse, xgboost_r2, xgboost_reg_report = regression_report(y_test, xgboost_y_pred)
# print(xgboost_reg_report)

# # Append all the statistics to the global lists
# all_mae.append(xgboost_mae)
# all_mse.append(xgboost_mse)
# all_rmse.append(xgboost_rmse)
# all_r2.append(xgboost_r2)
# all_models.append(xgboost_best_model)



# -----------------------------------------------------------------------------------------------------------------------

xgb2=xgboost_regressor.fit(X_train, y_train)
xgb_y_pred = xgb2.predict(X_test)
xgb_mae2, xgb_mse2, xgb_rmse2, xgb_r22, xgb_reg_report2 = regression_report(y_test, xgb_y_pred)
print(xgb_reg_report2)


########################################################################################################
#####                            SUPPORT VECTOR MACHINE REGRESSOR MODEL                            #####
########################################################################################################

# Initialize the Support Vector Machine Regressor
svr_regressor = SVR(random_state=21)

# Hyperparameters grid
svr_param_grid = {
    # The type of kernel to use
    'kernel': ['linear', 'poly', 'rbf'],
    # Regularization parameter (penalty for errors)
    'C': [0.1, 1, 10, 100],  
    # Epsilon parameter, controls margin of tolerance
    'epsilon': [0.01, 0.1, 0.2, 0.5],
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

# kernel: Specifies the type of kernel to use in the SVR:
# 'linear' for linear regression.
# 'poly' for polynomial kernel.
# 'rbf' (Radial Basis Function) for non-linear regression (most common).
# C: The regularization parameter. It trades off the model's ability to fit the training data versus its ability to generalize to new data:
# A higher C value means the model tries to fit the training data more closely (but might overfit).
# A lower C value allows more error but improves generalization.
# epsilon: The margin of tolerance within which no penalty is given for errors. A small epsilon value makes the model sensitive to the training data, while a larger value allows more margin for error.
# degree: The degree of the polynomial kernel function (only used when the kernel is 'poly'). Higher degrees can model more complex relationships, but may also increase the risk of overfitting.
# gamma: The kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. It determines the influence of a single training example on the decision boundary:
# 'scale' is 1 / (n_features * X.var()), which is generally a good default.
# 'auto' is 1 / n_features.
# A small value makes the model more general, while a larger value makes it more sensitive to individual points (increases complexity and risk of overfitting).
# tol: The tolerance for stopping criteria. Smaller values will make the algorithm run for longer but may result in more accurate results.
# cache_size: The size of the kernel cache in MB. A larger cache can speed up computation for large datasets but may use more memory.
# shrinking: Whether to use the shrinking heuristic to speed up training. When True, it can help speed up the convergence of the algorithm by reducing the number of support vectors considered at each step.


all_mae, all_mse, all_rmse, all_r2, all_models = regression_model(
    svr_regressor, svr_param_grid, kf, X_train, y_train, X_test, y_test, 
    all_mae, all_mse, all_rmse, all_r2, all_models)


# # Initialize GridSearchCV for hyperparameter tuning with CV
# # n_jobs=-1: using all processors
# # verbose=1: doesn't print anything (otherwise it prints a line for each combination of parameters tested and each CV folder)
# svr_grid_search = GridSearchCV(estimator = svr_regressor, param_grid = svr_param_grid, cv = kf, 
#                               scoring='neg_root_mean_squared_error', n_jobs = -1, verbose = 1)

# # Fit the model with GridSearchCV 
# svr_grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# print("Best Hyperparameters:", svr_grid_search.best_params_)

# # Get the best model from grid search
# svr_best_model = svr_grid_search.best_estimator_

# # Train the model
# svr_best_model.fit(X_train, y_train)

# # Predict on the test set
# svr_y_pred = svr_best_model.predict(X_test)

# # Evaluate the model on the test set
# svr_mae, svr_mse, svr_rmse, svr_r2, svr_reg_report = regression_report(y_test, svr_y_pred)
# print(svr_reg_report)

# # Append all the statistics to the global lists
# all_mae.append(svr_mae)
# all_mse.append(svr_mse)
# all_rmse.append(svr_rmse)
# all_r2.append(svr_r2)
# all_models.append(svr_best_model)





# -----------------------------------------------------------------------------------------------------------------------

svr2=svr_regressor.fit(X_train, y_train)
svr_y_pred2 = svr2.predict(X_test)
svr_mae2, svr_mse2, svr_rmse2, svr_r22, svr_reg_report2 = regression_report(y_test, svr_y_pred2)
print(svr_reg_report2)


########################################################################################################
#####                         COMPARE ALL THE MODELS AND SAVE THE BEST ONE                         #####
########################################################################################################

# Get the best model (with the lowest RMSE)
index_min_mae = all_mae.index(min(all_mae))
index_min_mse = all_mse.index(min(all_mse))
index_min_rmse = all_rmse.index(min(all_rmse))
index_min_r2 = all_r2.index(max(all_r2))
all_models[index_min_mae],all_models[index_min_mse],all_models[index_min_rmse],all_models[index_min_r2]
best_model = all_models[index_min_rmse]

# Save the model (and all the encoders)
saved_model = {"model": best_model,  "gender_encoder": gender_encoder, "workout_type_encoder": workout_type_encoder}
with open('C:/Users/Admin/Documents/Python Projects/gym_data_analysis/best_model.pkl', 'wb') as file:
    pickle.dump(saved_model,file)

