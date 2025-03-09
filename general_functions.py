# Load the libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy import stats
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV


# Create the function to display a barplot (data viz for categorical variable)
def barplot_visualization(data,variable):
    """
    Display a barplot for the specific variable
    
    Arguments:
    data -- dataframe
    variable -- categorical variable (string)
    """
    # Compute the frequency and percentage for each class of the categorical variable and sort by class
    data_plot = data[variable].value_counts().reset_index()
    data_plot['percentage']=np.round(data_plot['count']*100/data_plot['count'].sum(),2)
    data_plot=data_plot.sort_values(by=variable,ignore_index=True)

    # Create a barplot for the variable proportions
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data_plot, x=variable, y='count')

    # Add percentage labels
    for index, row in data_plot.iterrows():
        plt.text(
            index, 
            # Slightly under the bar
            row['count'] - row['count']/6,
            # Format the percentage
            f'{row["percentage"]:.2f}%',
            ha='center', 
            fontsize=10
    )
        
    # Add the main title, title for each axis, and display the categories label horizontally
    plt.title('Proportion per '+variable.replace('_',' ')+' in the data')
    plt.xlabel(variable.capitalize().replace('_',' '))
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    # Remove legend
    plt.legend([], [], frameon=False)
    plt.show()


# Create the function to display a boxplot (data viz for numerical variable)
def boxplot_visualization(data,variable,new_variable):
    """
    Display the barplot for the specific variable(s)
    
    Arguments:
    data -- dataframe
    variable -- list of numerical variable(s), eventually several ones (string)
    new_variable -- new variable name, especially used when we plot the boxplot of different variables in 1 plot (string)
    """
    # Create the melted data (especially used when we plot the boxplot of different variables in 1 plot)
    melted_data = pd.melt(data[variable])
    melted_data['variable'] = melted_data['variable'].str.replace('_', ' ').str.capitalize()

    # Build and display the boxplot
    plt.figure(figsize=(3,5))
    sns.boxplot(y='value', x='variable', data=melted_data)
    plt.title("Distribution of "+' & '.join(variable).replace('_',' ')+' in the data')
    plt.ylabel(new_variable)
    plt.xlabel('')
    plt.show()


# Create the function to display the density plot (data viz for numerical variable)
def kdeplot_visualization(data,variable,new_variable):
    """
    Display the density plot for the specific variable(s)
    
    Arguments:
    data -- dataframe
    variable -- list of numerical variable(s), eventually several ones (string)
    new_variable -- new variable name, especially used when we plot the boxplot of different variables in 1 plot (string)
    """
    # Build the density plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data[variable], fill=True)

    # Set up the main title and title for each axis
    plt.title('Density plot of '+' & '.join(variable).replace('_',' ')+' in the data')
    plt.xlabel(new_variable)
    plt.ylabel('Density')
    plt.show()


# Create the function to display the density plot for each class of a categorical avriable
def bi_kdeplot_visualization(data, categorical_variable, numerical_variable):
    """
    Display the density plot for each category of the specific categorical variable
    
    Arguments:
    data -- dataframe
    categorical_variable -- categorical variable (string)
    numerical_variable -- numerical variable (string)
    """
    # Build the density plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data, x=numerical_variable, hue=categorical_variable, fill=True) #,color='Set1'

    # Set up the main title and title for each axis
    plt.title('Density plot of '+numerical_variable.replace('_',' ')+' per '+categorical_variable+' in the data')
    plt.xlabel('')
    plt.ylabel(numerical_variable.replace('_','').capitalize())
    plt.show()


# Create the function to display the univariate analysis for a numerical variable (missing value, boxplot, describe, density plot)
def numerical_univariate_analysis(data,variable,new_variable):
    """
    Implement the univariate analysis for a numerical variable (missing value, boxplot, describe, density plot)
    
    Arguments:
    data -- dataframe
    variable -- list of numerical variable(s), eventually several ones (string)
    new_variable -- new variable name, especially used when we plot the boxplot of different variables in 1 plot (string)
    """
    # Check for missing values
    count_na = data[variable].isna().sum()
    for column, count in count_na.items():
        print(f'Number of missing values for variable {column}: {count}')

    # Display the boxplot for the variable
    boxplot_visualization(data,variable,new_variable)

    # Check for the main statistics of the variable
    variable_describe = data[variable].describe()
    print('Describe of variable '+' & '.join(variable)+': ')
    print(variable_describe)

    # Display the density plot
    kdeplot_visualization(data,variable,new_variable)


# Create the function to display the univariate analysis for a categorical variable (missing value, barplot, describe)
def categorical_univariate_analysis(data,variable):
    """
    Implement the univariate analysis for a categorical variable (missing value, barplot, describe)
    
    Arguments:
    data -- dataframe
    variable -- categorical variable (string)
    """
    # Check for missing values
    count_na = data[variable].isna().sum()
    print(f'Number of missing values for variable {variable}: {count_na}')

    # Display for the barplot
    barplot_visualization(data,variable)


# Create the function to implement the bivariate analysis between 2 numerical variables (kendall correlation and scatterplot)
def numerical_numerical_bivariate_analysis(data,X_variable,Y_variable):
    """
    Create the bivariate analysis between 2 numerical variables (kendall correlation and scatterplot)
    
    Arguments:
    data -- dataframe
    X_variable -- numerical variable, part of the feature variables (string)
    Y_variable -- numerical variable, target variable (string)
    """
    # Compute the correlation between both variables
    # method='kendall' because Pearson only check the linear relationship
    corr = data[Y_variable].corr(data[X_variable],method='kendall')
    corr = np.round(corr,2)
    print('Correlation between '+X_variable+' and '+Y_variable+' is: '+ str(corr))

    # Display the scatterplot between both variables
    sns.set_theme(style="ticks")
    plt.scatter(data[X_variable], data[Y_variable], alpha=0.5)
    plt.title('Scatterplot of ' + Y_variable.replace('_',' ') + ' vs ' + X_variable.replace('_',' '))
    plt.xlabel(X_variable.replace('_',' ').capitalize())
    plt.ylabel(Y_variable.replace('_',' ').capitalize())
    plt.tight_layout()
    plt.show()


# Create the function to perform the Levene test
def levene_test(data, X_variable, Y_variable):
    """
    Implement the Levene test. 
    Levene test aims at controlling if the variance of Y variable for each category of the X variable is the same 

    Arguments:
    data -- dataframe
    X_variable -- categorical variable (string)
    Y_variable -- numerical variable (string)
    
    Returns:
    f_stat -- F-statistics of the Levene test
    p_value -- pvalue  of the Levene test
    """
    # Get the unique categories of the X variable
    X_variable_groups = data[X_variable].unique()
    group_dict = {}

    # Gather the Y variables for each category of X variable in a dictionnary
    for v in X_variable_groups:
        group_name = 'group_' + str(v)
        group_dict[group_name] = data.loc[data[X_variable] == v, Y_variable]
    
    # Perform Levene test
    # *: to unpack the values of the dictionnary as arguments
    f_stat, p_value = stats.levene(*group_dict.values()) 
    
    return [f_stat, p_value]


# Create the function to implement the T-test
def t_test(data, X_variable, Y_variable, var_equal):
    """
    Implement the T-test (when X variable has 2 categories). 
    T-test aims at controlling if the mean of Y variable for each category of the X variable is the same.

    Arguments:
    data -- dataframe
    X_variable -- categorical variable (string)
    Y_variable -- numerical variable (string)
    var_equal - boolean (True or False) saying if the variance of Y variable for each category of X variable are equal or not
    
    Returns:
    f_stat -- F-statistics of the T-test
    p_value -- pvalue  of the T-test
    """
    # Get the unique categories of the X variable
    X_variable_groups = data[X_variable].unique()
    group_dict = {}

    # Gather the Y variables for each category of X variable in a dictionnary
    for v in X_variable_groups:
        group_name = 'group_' + str(v)
        group_dict[group_name] = data.loc[data[X_variable] == v, Y_variable]
    
    # Perform T-test
    # *: to unpack the values of the dictionnary as arguments
    f_stat, p_value = stats.ttest_ind(*group_dict.values(), equal_var=var_equal, alternative='two-sided')
    
    return [f_stat, p_value]



# Create the function to implement the ANOVA test
def anova_test(data, X_variable, Y_variable):
    """
    Implement the ANOVA test (when X variable has more than 2 categories and variance of Y among categories of X are the same). 
    ANOVA test aims at controlling if the mean of Y variable for each category of the X variable is the same.

    Arguments:
    data -- dataframe
    X_variable -- categorical variable (string)
    Y_variable -- numerical variable (string)
    
    Returns:
    f_stat -- F-statistics of the ANOVA test
    p_value -- pvalue  of the ANOVA test
    """
    # Get the unique categories of the X variable
    X_variable_groups = data[X_variable].unique()
    group_dict = {}

    # Gather the Y variables for each category of X variable in a dictionnary
    for v in X_variable_groups:
        group_name = 'group_' + str(v)
        group_dict[group_name] = data.loc[data[X_variable] == v, Y_variable]

    # Perform ANOVA test
    # *: to unpack the values of the dictionnary as arguments
    f_stat, p_value = stats.f_oneway(*group_dict.values())

    return [f_stat, p_value]


# Create the function to implement the Kruskal-Wallis test
def kruskal_wallis_test(data, X_variable, Y_variable):
    """
    Implement the Kruskal-Wallis test (when X variable has more than 2 categories and variance of Y among categories of X are not the same). 
    Kruskal-Wallis test aims at controlling if the mean of Y variable for each category of the X variable is the same.

    Arguments:
    data -- dataframe
    X_variable -- categorical variable (string)
    Y_variable -- numerical variable (string)
    
    Returns:
    f_stat -- F-statistics of the Kruskal-Wallis test
    p_value -- pvalue  of the Kruskal-Wallis test
    """
    # Get the unique categories of the X variable
    X_variable_groups = data[X_variable].unique()
    group_dict = {}

    # Gather the Y variables for each category of X variable in a dictionnary
    for v in X_variable_groups:
        group_name = 'group_' + str(v)
        group_dict[group_name] = data.loc[data[X_variable] == v, Y_variable]
    
    # Perform Kruskal-Wallis test
    # *: to unpack the values of the dictionnary as arguments
    f_stat, p_value = stats.kruskal(*group_dict.values())

    return [f_stat, p_value]


# Create the function to implement the bivariate analysis between 1 numerical variable and 1 categorical variable 
def numerical_categorical_bivariate_analysis(data, categorical_variable, numerical_variable):
    """
    Create the bivariate analysis between a numerical variable and a categorical variable
    * boxplot of the numerical variable for each class of the categorical variable
    * density plot of the numerical variable for each class of the categorical variable
    * Levene test to check if variance of Y is the same for each category of X
    * T-test to check the significant link between Y and X (with 2 classes)
    * ANOVA test to check the significant link between Y and X (with more than 2 classes and same variance of Y among groups)
    * Kruskal-Wallis test to check the significant link between Y and X (with more than 2 classes and different variance of Y among groups)
    
    Arguments:
    data -- dataset
    categorical_variable -- categorical variable (string)
    numerical_variable -- numerical variable (string)
    """
    # Display the boxplot of Y for each class of x
    plt.figure(figsize=(3, 5))
    sns.boxplot(y=data[numerical_variable], hue=data[categorical_variable], gap=0.2)
    plt.title("Distribution of the amount of "+numerical_variable.replace('_',' ')+" per "+categorical_variable.replace('_',' ')+" in the data")
    plt.ylabel(numerical_variable.replace('_',' ').capitalize())
    plt.xlabel(categorical_variable.replace('_',' ').capitalize())
    # Remove x-axis ticks
    plt.xticks([])  
    plt.legend(title=categorical_variable.replace('_',' ').capitalize())
    plt.show()

    # Display the density plot to check the normality of the distribution assumption
    bi_kdeplot_visualization(data, categorical_variable, numerical_variable)

    # Levene test (to check if variances of numerical_variable are the same in all groups of categorical_variable)
    levene_test_pvalue = levene_test(data, categorical_variable, numerical_variable)[1]
    print('Levene test pvalue is: '+str(np.round(levene_test_pvalue,2)))
    if levene_test_pvalue < 0.05:
        var_equal = False
    else:
        var_equal = True

    # If categorical_variable has 2 classes, T-test
    if len(data[categorical_variable].unique())==2:
        # T-test
        t_test_pvalue = t_test(data,categorical_variable,numerical_variable,var_equal)[1]
        print('T-test pvalue is: '+str(np.round(t_test_pvalue,2)))
    else:
        # If categorical_variable has more than 2 classes and variances are the same, ANOVA test
        if (var_equal==True):
            anova_test_pvalue = anova_test(data,categorical_variable,numerical_variable)[1]
            print('ANOVA test pvalue is: '+str(np.round(anova_test_pvalue,2)))
        # If categorical_variable has more than 2 classes and variances are not the same, Kruskal-Wallis test
        else:
            kruskal_wallis_test_pvalue = kruskal_wallis_test(data,categorical_variable,numerical_variable)[1]
            print('Kruskal-Wallis test pvalue is: '+str(np.round(kruskal_wallis_test_pvalue,2)))


# Function to create a regression report (display the main statistics for a regression project: MAE, MSE, RMSE, R2)
def regression_report(y_true, y_pred):
    """
    Create a regression report (with the main statistics for a regression project: MAE, MSE, RMSE, R2)
    
    Arguments:
    y_true -- vector of true values for y
    y_pred -- vector of predicted values for y
    
    Returns:
    mae -- MAE between y_true and y_pred
    mse -- MSE between y_true and y_pred
    rmse --RMSE between y_true and y_pred
    r2 -- R2 of the model (% of variance explained by the model)
    reg_report -- string summing up all the statistics
    """
    # Compute the Mean Absolute Error between y_true and y_pred
    mae = metrics.mean_absolute_error(y_true, y_pred) 

    # Compute the Mean Squared Error between y_true and y_pred
    mse = metrics.mean_squared_error(y_true, y_pred)
    
    # Compute the Root Mean Squared Error between y_true and y_pred
    rmse = np.sqrt(mse)
    
    # Compute the R2 of the model
    r2 = metrics.r2_score(y_true, y_pred)
    
    # Define the regression report pasting all statistics into 1 string
    reg_report = 'R2: ' + str(round(r2*100,2)) + '%\nMAE: ' + str(round(mae,2)) + '\nMSE: ' + str(round(mse,2)) + '\nRMSE: '+str(round(rmse,2))

    return(mae, mse, rmse, r2, reg_report)


# Function to create a regression model (get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models)
def regression_model(regressor, param_grid, kf, X_train, y_train, X_test, y_test, all_mae, all_mse, all_rmse, all_r2, all_models):
    """
    Create a function to get the best parameters, train and evaluate the model, save the main statistics in a list to compare with all models
    
    Arguments:
    regressor -- regressor of the model we want to train
    param_grid -- parameters grid used for the GridSearchCV
    kf -- kfold CV
    X_train -- dataset (features) used to train the model
    y_train -- target used to train the model
    X_test -- dataset (features) used to evaluate the model
    y_test -- target used to evaluate the model
    all_mae -- list gathering the MAE of the previous trained models (to compare all models later and choose the best one)
    all_mse -- list gathering the MSE of the previous trained models (to compare all models later and choose the best one)
    all_rmse -- list gathering the RMSE of the previous trained models (to compare all models later and choose the best one)
    all_r2 -- list gathering the R2 of the previous trained models
    all_models -- list gathering all the previous models
    
    Returns:
    all_mae -- list gathering the MAE of all the models trained until now (previous one + current one)
    all_mse -- list gathering the MSE of all the models trained until now (previous one + current one)
    all_rmse -- list gathering the RMSE of all the models trained until now (previous one + current one)
    all_r2 -- list gathering the R2 of all the models trained until now (previous one + current one)
    all_models -- list gathering all the models trained until now (previous one + current one)
    """
    # Initialize GridSearchCV for hyperparameter tuning with CV
    # n_jobs=-1: using all processors
    # verbose=1: doesn't print anything (otherwise it prints a line for each combination of parameters tested and each CV folder)
    grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid, cv = kf, 
                              scoring='neg_root_mean_squared_error', n_jobs = -1, verbose = 1)

    # Fit the model with GridSearchCV 
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Train the model
    best_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model on the test set
    mae, mse, rmse, r2, reg_report = regression_report(y_test, y_pred)
    print('Regression report on the test set:')
    print(reg_report)

    # Append all the statistics to the global lists
    all_mae.append(mae)
    all_mse.append(mse)
    all_rmse.append(rmse)
    all_r2.append(r2)
    all_models.append(best_model)

    return(all_mae, all_mse, all_rmse, all_r2, all_models)
