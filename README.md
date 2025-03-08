# Data project about calories burned in the gym


## Project description
The goal of this project is to carry on an exploratory data analysis (EDA) to analyse gym data, then to modelize the calories burned at gym.
The project contains 2 main parts:
* Exploratory Data Analysis
* Modeling the calories burned


## Dataset
We use the gym dataset that we downloaded from Kaggle (https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset).
It contains 973 rows and 15 variables, the following ones: 
* Age: age of the gym member
* Gender: gender of the gym member (Male or Female)
* Weight (kg): weight of the gym member (in kilograms)
* Height (m): height of the gym member (in meters)
* Max_BPM: maximum heart rate (beats per minute) of the gym member during the workout session
* Avg_BPM: average heart rate (beats per minute) of the gym member during the workout session
* Resting_BPM: heart rate (beats per minute) of the gym member before the workout session
* Session_Duration (hours): duration of the workout session in hours
* Calories_Burned: calories burned by the gym member during the workout session
* Workout_Type: type of workout performed during the session by the gym member (Cardio, HIIT, Strength or Yoga)
* Fat_Percentage: body fat percentage of the gym member
* Water_Intake (liters): water intake during workouts by the gym member
* Workout_Frequency (days/week): number of workout sessions per week
* Experience_Level: level of experience of the gym member (1 for beginner, 2 for medium and 3 for expert)
* BMI: Body Mass Index computed as the weight divided by the squared height 


## Exploratory Data Analysis

### Univariate data analysis
The univariate data analysis allows to investigate each variable 1 by 1. During this step, we can check for missing values and outliers for each variable. This step also allows to better understand the data, checking each variable and looking at their distribution (skewness or similarity).
Insights we get from the univariate data analysis:
* gender: 
    * this variable doesn't contain any missing values, 
    * there is no outliers: we only have 2 different values Female and Male 
    * we notice that the proportion of men vs women is pretty similar (47% of women vs 53% of men)
* age: 
    * this variable doesn't contain any missing values, 
    * we don't see any outliers either
    * we notice that more than 50% are between 28 and 49yo (I was rather expecting people to be mostly between 25 and 35)
* BPM (resting BPM, average BPM, max BPM): 
    * this variable doesn't contain any missing values, 
    * we don't see any outliers either
    * we notice (as expected) that resting bpm < average bpm < max bpm. Moreover, we notice that the distribution of average bpm and max bpm is less different than with resting bpm, meaning that the bpms of people are less different when people are rested than when do sport or bpms of people are more different when they do sport (probably because level of sport is different based on people and in general people spend more time resting than exercising)
    * the density distributions are pretty symetric, almost uniform law
* weight: 
    * this variable doesn't contain any missing values, 
    * we see a couples of outliers at the top
    * we notice that 50% of the gym members (from this dataset) have a weight between 58.1 and 86kg. Moreover, the distribution is not so symmetric, we observer a positive/right skewness
* height: 
    * this variable doesn't contain any missing values, 
    * we don't see any outliers either
    * we notice that 50% of the gym members in this dataset have a height between 1.62 and 1.8m. Moreover, the distribution is pretty much symmetric, slightly more weight in the lower heights
* session_duration: 
    * this variable doesn't contain any missing values, 
    * we don't see any outliers either
    * we notice that more than 50% of the gym members spend between 1h and 1.5h at the gym. We observer a distribution very symetric
* calories_burned: 
    * this variable doesn't contain any missing values, 
    * we see a couple of outliers at the top
    * we notice that more than 50% of the gym members are burning between 720 and 1080 calories. Moreover we observe a distribution rather symetric (specially if we don;t check the outliers)
* workout_type: 
    * this variable doesn't contain any missing values, 
    * there is no outliers: we only have 4 different values (Cardio, HIIT, Strenght and Yoga)
    * we notice that the frequency between the 4 sports is similar: HIIT is the less represented (23%) and strength is the most represented (27%)
* fat_percentage: 
    * this variable doesn't contain any missing values, 
    * we don't see any outliers either
    * we notice that more than 50% of the gym members in this dataset have a fat percentage between 21% and 30% (COMPARISON WITH OTHER PEOPLE). Moreover, we observe that the distribution is not so symetric, it's rather negative skewed
* water_intake: 
    * this variable doesn't contain any missing values, 
    * we don't see any outliers either
    * we notice that more than 50% of the gym members are drinking between 2,2 and 3.1 liters of water. Moreover, we observe that the distribution is rather symetric, except a big down on 3 (2 modes)
* workout frequency: 
    * this variable doesn't contain any missing values, 
    * there is no outliers: we only have 4 different values (2 to 5)
    * we notice that nearly 70% of the gym members go between 3 and 4 times
* experience level: 
    * this variable doesn't contain any missing values, 
    * there is no outliers: we only have 3 different values (1 to 3 for beginner to expert)
* BMI: 
    * this variable doesn't contain any missing values, 
    * we see a couple of outliers at the top
    * we notice that more than 50% have a BMI between 20.1 and 28.6 (COMPARISON WITH NORMAL PEOPLE). We observe that the distribution is rather symetric if we don't check the outliers. With the outliers, the distribution is rather positive skewed

With this univariate data analysis, we can conclude that:
* the data has no missing value
* the data has a couple of outliers that will be handled in the cleaning step


### Cleaning step
The data has no missing value, so we don't need to carry a data imputation step. If we had missing data, we should have done the following steps:
* imputation for numeric variable: impute with the mean or the median (median is more robust to outliers)
* imputation for categoric variable: impute with the mod
We only see a couple of outliers but they are not that extreme, so we will just leave them, If we are in the case of an extreme value, if the percentage of outliers is very small, we should delete them. Otherwise, if we need to need to impute them etc.


### Bivariate data analysis
We are now doing bivariate data analysis (analysis each variate with respect to the burned calories [the variable we will want to predict]). Burned calories variable is numeric to we will carry out 3 types of asociation:
* numeric to numeric relationship: for each numeric variable, we will draw the scatter plot between this variable and the calories burned variable. We will also compute the correlation coefficient between this variable and the calories burned variable (with Kendall method because Pearson only catches the linear relationship). This is what we noticed about the relationship between calories burned and each of the following variable (based on the scatterplot and the correlation):
    * age: no strong association (cor = -0.1, negative correlation meaning the older, the elss calories are burned, indeed the body metabolism is less active the older we get [but might also comes from other variables])
    * weight: no strong association (cor = 0.09)
    * height: no strong association (cor = 0.05)
    * max_bpm: no strong association (cor = -0.01)
    * avg_bpm: some association (cor = 0.23, positive correlation meaning the higher is the BPM on average during the session, the more calories are burnt [this is well known, fast heart rate makes the calories being burnt])
    * resting_bpm: no strong association (cor = -0.0)
    * session_duration: strong (positive) association (cor = 0.74, strong positive correlation meaning the longer is the workout session, the more calories are burnt)
    * fat_percentage: strong (negative) association (cor = -0.35)
    * water_intake: some association (cor = 0.24, positive correlation meaning the more water the member has drunk during the day, the more calories is burnt)
    * bmi: no strong association (cor = 0.05)
* numeric to categoric relationship: for each categoric variable, we:
    * draw the boxplot (of the calories burned variable for each class of the categoric variables): to check the distribution of the calories burned for each class
    * draw the distribution plot (calories burned for each class of the categoric variables): to check if the distribution is somewhat normal (the sample is pretty large anyway)
    * carry on the Levene test: to check if the variance of Y is the same among the class of X
    H0: variances of calories_burned on both groups are the same
    H1: variances of calories_burned on both groups are different
    If pvalue < 0.05, we reject H0: variances of Y among X are different
    * carry on the T-test: to test for significant link between X and Y (if X has only 2 classes), it determines if there's a significant difference between the means of 2 groups.
    H0: the true means of Y among different classes of X are equal
    H1: the true mean of Y among the different groups of X is different
    If p-value< 0.05, we reject H0, meaning that the true means of Y is significantly different for the different groups of X
    * carry on ANOVA test: to test for significant link between X and Y (if X has more than 2 classes and variance of Y is the same among classes of X). It tests if mean (numerical) over categories (categorial with 3+ groups) is the same. Assumptions: normality of data (checked by density plot), independence of observations (in this case it's always the case, records can't be in both groups), homogeneity of variance (we checked it with Levene test, otherwise we carry on a Kruskal-Wallis test)
    H0: the mean of variable Y across the categories of variable X are equal (no difference in means)
    H1: at least one groups mean significantly differs
    If pvalue < 0.05, we reject H0 and conclude that at least one group has a significantly different mean and we conclude that variable X significantly affects variable Y
    * carry on Kruskal-Wallis test: to test for significant link between X and Y (if X has more than 2 classes and variance of Y is not the same among classes of X)
    H0: means are the same
    H1: means are different
    If pvalue < 0.05, we reject H0: mean of calories burned is different depending on the groups of X
    This is what we notice about the relationship between calories burned and each of the following variables (based on the boxplot, distribution plot and the different test):
    * gender: from the boxplot, we notice that male burn more calories than woman (but it can be linked to other variables, for example men are training longer or harder or are more expert). 
    From the density plot, we notice that the distribution of Y is normal for both classes. We assume normality since we have a pretty big dataset and the density plot showed us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue < 0.05 so we reject H0: variance of Y is different for both gender groups.
    We carry on a 2 sample T-test (since gender has only 2 classes: Male and Female) with unequal variances. T-test pvalue < 0.05 so we reject H0, the true means of calories_burned among female and male are different.
    * workout_type: from the boxplot, we don;t see so much different in the calories_burned depending on the sport. 
    From the density plot, we notice that the distribution of Y is normal for all classes. We assume normality since we have a pretty big dataset and the density plot showed us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue > 0.05 so we don't reject H0: variance of Y is the same for all workout type groups.
    We carry on an ANOVA test (since workout type has more than 2 classes: Cardio, HIIT, Strength and Yoga) with equal variances. ANOVA pvalue > 0.05 so we don't reject H0, mean of calories burned is the same for the 4 different workout types.
    * experience_level: from the boxplot, we do see difference in the calories_burned depending on the experience level. The more experienced the member is, the more calories the member will burn. It's pretty straightforward: expert burn more, probably because they train harder or know what to train to burn calories
    From the density plot, we notice that the distribution of Y is normal for all classes. We assume normality since we have a pretty big dataset and the density plot showed us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue < 0.05 so we reject H0: variance of Y are not the same for all groups of experience_level, so we can't use ANOVA test anymore.
    We carry on a Kruskal-Wallis test (since experience level has more than 2 classes: 1, 2 and 3) with unequal variances. Kruskal-Wallis pvalue <0.05 so we reject H0: mean of calories burned is different depending on the experience level
    * workout_frequency: from the boxplot, we do see difference in the calories_burned depending on the workout_frequency. The more often the member goes to the gym, the more calories the member will burn. It's pretty straightforward: people who come often to the gym burn more, probably because they are more expert
    From the density plot, we notice that the distribution of Y is normal for all classes. We assume normality since we have a pretty big dataset and the density plot showed us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue < 0.05 so we reject H0: variance of Y are not the same for all groups of workout_frequency, so we can't use ANOVA test anymore.
    We carry on a Kruskal-Wallis test (since workout_frequency has more than 2 classes:  2, 3, 4 and 5) with unequal variances. Kruskal-Wallis pvalue < 0.05 so we reject H0: mean of calories burned is different depending on the workout_frequency

To sum up, we find the stronger relationship between calories_burned variable and the following (X) variables: avg_bpm, session_duration, fat_percentage and water_intake, gender, experience_level and workout_frequency


## Modeling the calories burned
We want to predict the calories burned at the gym using all other variables. We use several models and we compare which models give the best results.

### Data processing
Before modeling we do a couple of actions:
* transform all variables to numeric:
    * for ordinal variables (experience level and workout frequency): we just switch the type to integer
    * for nominal variables (gender and workout type): we encode the variables usig LabelEncoder for gender (from Female/Male to 0/1) and using OneHotEncoder for workout type (from Cardio to [1 0 0], from HIIT to [0 1 0], from Strength to [0 0 1] and from Yoga to [0 0 0])
* split X (input features) and y (target labels)
* split the dataset into train and test sets: we keep 20% for the test set
* normalize the train set then the test set to avoid having the model being mainly influenced by a set of variables only because they have a bigger scale (some models are very sensible to this like KNN).

### Models
We use several models to modelize y with respect to X and choose the best one. 
To train the models, we use cross-validation, it allows to split the training set in smaller samples to have more training samples and the model is trained several times (for each CV sample).
For each model, we set up an hyperparameters grid search, the model will train for each combination of parameters and detemine the best parameters for this model. The number of possible candidates is determined by multiplying the number of options for each parameter (and finally multiply by the number of folders from the cross-validation).

Used models:
* Decision trees: the algorithm splits the data into subsets based on feature values, creating a tree-like structure of decisions (made with nodes, branches and leaves). At the beginning, the root node contains the whole dataset. Then at every step (at each decision node), the tree splits the data from the node based on a condition on a feature to maximize the purity of the subsets. The leaves contain the predicted output (mean in case of a regression). It is a simple, interpretable machine learning algorithm but it is sensitive to small changes in data, very easy to overfit (especially with deep trees). To make a prediction, the tree follows the splits from the root to a leaf node. We use the following parameters to train a decision tree model:
    * max_depth: the maximum depth of the tree (limiting the depth can help prevent overfitting by controlling the complexity of the model)
    * min_samples_split: the minimum number of samples required to split an internal node (larger values prevent the model from creating nodes that are too specific to the training data)
    * min_samples_leaf: the minimum number of samples required to be at a leaf node (increasing this value can smooth the model, making it more generalized)
    * max_features: the number of features to consider when looking for the best split. Several options are possible: all features (None), the square root of the number of features ('sqrt'), or the log base 2 of the number of features ('log2')
    * splitter: the strategy used to split at each node. Several options are possible: 'best' (to choose the best split) or ('random' to choose a random split).
    * max_leaf_nodes: the maximum number of leaf nodes in the tree (limiting this helps prevent overfitting by restricting the complexity of the tree)
* K-Nearest Neighbor: the algorithm makes predictions based on the k closest data points (neighbors) in the training set. For regression, the predicted value is the average (or weighted average) of the k nearest neighbors' target values. KNN uses a distance metric (commonly Euclidean distance) to find the closest neighbors. It's a simple, easy to understand and effective (with small and medium datasets) algorithm but it's computationally expensive (especially for large datasets). We use the following parameters to train a KNN model:
    * n_neighbors: the number of neighbors to use when making predictions (lower values may lead to more variance, and higher values may lead to more bias)
    * weights: defines how the distance to neighbors influences the prediction ('uniform' gives all neighbors equal weight, while 'distance' gives closer neighbors more weight)
    * algorithm: the algorithm to compute the nearest neighbors ('auto' chooses the best algorithm based on the data)
    * metric: the distance metric to use for finding neighbors (common options include 'euclidean' and 'manhattan')
* Random Forest: it's an ensemble learning algorithm that combines multiple decision trees to improve predictive performance. To make a prediction (in case of regression like here) it averages the predictions from all the trees. Each tree is built using bootstrapped samples (random subsets with replacement) from the training data, and at each split, a random subset of features is considered to prevent overfitting. This algorithm reduces overfitting compared to individual decision trees, it handles large datasets well and it can capture complex relationships in the data but 
it's more computationally expensive and less interpretable than a single decision tree. We use the following parameters to train a random forest model:
    * n_estimators: the number of trees in the forest (a higher number generally leads to a better model, but it also increases computational cost)
    * max_depth: the maximum depth of the tree
    * min_samples_split: the minimum number of samples required to split an internal node.
    * min_samples_leaf: the minimum number of samples required to be at a leaf node.
    * max_features: the number of features to consider when looking for the best split (common options include 'auto' (all features), 'sqrt' (square root of the number of features) or an integer)
    * bootstrap: whether to use bootstrap samples when building trees (set to True to sample with replacement)
    * warm_start: if True, it reuses the solution of the previous fit and adds more trees, which can speed up computation
* Gradient Boosting Regressor: this algorithm is an ensemble learning technique. It builds a model by combining multiple weak learners in a sequential manner (one at a time), where each tree tries to correct the errors (residuals) of the previous one. The "gradient" in Gradient Boosting refers to using gradient descent to minimize the loss (error) function by adjusting the weights of the trees. This model often achieves high accuracy and is robust to overfitting with proper tuning but it's computationally expensive and slower to train compared to other models, it's sensitive to noisy data and outliers and it requires careful tuning of hyperparameters (e.g., learning rate, number of estimators). We use the following parameters to train a GB model:
    * n_estimators: the number of boosting stages (trees) to fit (more trees can lead to a more complex model)
    * learning_rate: the step size to shrink the contribution of each tree (smaller values may lead to better performance but require more trees)
    * subsample: the fraction of samples used for fitting each tree (lower values help in reducing overfitting)
    * max_depth: the maximum depth of the individual decision trees (deeper trees can model more complex relationships)
    * min_samples_split: the minimum number of samples required to split an internal node (higher values prevent the model from learning overly specific patterns)
    * min_samples_leaf: the minimum number of samples required to be at a leaf node (this helps in smoothing the model)
    * max_features: the number of features to consider when looking for the best split (to be tune to prevent overfitting and speed up the model)
* Extreme Gradiant Boosting: Extreme Gradient Boosting (XGBoost) is an optimized, highly efficient and fast version of the Gradient Boosting model. It’s designed to perform better in terms of speed, accuracy, and scalability.
XGBoost builds trees sequentially, where each tree corrects the errors of the previous one. XGBoost includes L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting, making it more robust compared to traditional Gradient Boosting. XGBoost supports parallel processing during training, making it faster than standard gradient boosting algorithms. XGBoost uses a more advanced tree pruning technique (post-pruning) that helps reduce the complexity of the model while improving performance. It can hard to interpret (like most tree0based models) compared to simpler models. We use the following parameters to train a XGBoost model:
    * n_estimators: the number of boosting rounds (trees) (more trees generally lead to a better model, but too many can overfit)
    * learning_rate: determines how much each tree contributes to the final model (smaller values are typically preferred as they slow down the learning process, which may improve performance but requires more trees)
    * max_depth: controls the depth of each tree (larger depths can capture more complex relationships but may also overfit)
    * min_child_weight: controls overfitting by determining the minimum sum of instance weights in a child (larger values prevent the model from learning overly specific patterns)
    * subsample: fraction of the data to use for training each tree (by setting it under 1, some randomness is introduced into the model and help reduce overfitting)
    * colsample_bytree: fraction of features to use for each tree (by setting it under 1, some randomness is introduced into the model and help reduce overfitting)
    * gamma: controls the regularization of the tree (higher values make the algorithm more conservative  by preventing further splits)
    * reg_alpha: L1 (Lasso) regularization term on weights (it helps in controlling the overfitting)
    * reg_lambda: L2 (Ridge) regularization term on weights (it helps in controlling the overfitting)
    * objective: loss function ('reg:squarederror' is the most ommon for regression task)
    * booster: type of boosting model. Several options are possible: 'gbtree' (tree-based), 'gblinear' (linear so rather faster but may not perform so well on non-linear problems), or 'dart' (Dropout trees, a regularization technique that can help with overfitting)



* Support Vector Regressor: it aims at finding the best hyperplane that approximates the data while maintaining a margin of tolerance for errors (up to a certain threshold epsilon). Data points within this margin are considered as correct. SVR can use different kernel functions (linear, polynomial, radial basis function, etc.) to map data into higher-dimensional spaces, allowing it to capture non-linear relationships. Only a subset of the data points, called support vectors, influence the model. These points are the ones closest to the margin and play a key role in defining the model. SVR is an effective model for both linear and non-linear regression tasks, it's robust to overfitting, especially with appropriate kernel selection and regularization but it can be computationally expensive, especially for large datasets and the choice of kernel, regularization, and epsilon values can be tricky to tune. We use the following parameters to train a SVR model:
    * kernel: specifies the type of kernel to use in the SVR. Several options are possible: 'linear' (for linear regression), 'poly' (for polynomial kernel), 'rbf' (Radial Basis Function, for non-linear regression, [most common])
    * C: the regularization parameter (penalty for errors). It trades off the model's ability to fit the training data versus its ability to generalize to new data (a higher C value means the model tries to fit the training data more closely [but might overfit], while a lower C value allows more error but improves generalization)
    * epsilon: the margin of tolerance within which no penalty is given for errors (a small epsilon value makes the model sensitive to the training data, while a larger value allows more margin for error)
    * degree: the degree of the polynomial kernel function (only used when the kernel is 'poly') (higher degrees can model more complex relationships, but may also increase the risk of overfitting)
    * gamma: the kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels, it determines the influence of a single training example on the decision boundary. Several options are possible: 'scale' which is 1 / (n_features * X.var()) (generally a good default), 'auto' which is 1 / n_features, a number (a small value makes the model more general, while a larger value makes it more sensitive to individual points and increase complexity and risk of overfitting)
    * tol: the tolerance for stopping criteria (smaller values will make the algorithm run for longer but may result in more accurate results)
    * cache_size: the size of the kernel cache in MB. A larger cache can speed up computation for large datasets but may use more memory
    * shrinking: whether to use the shrinking heuristic to speed up training. When True, it can help speed up the convergence of the algorithm by reducing the number of support vectors considered at each step.


