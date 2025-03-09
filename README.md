# Data project about calories burnt in the gym

## Project description
The goal of this project is to carry on an exploratory data analysis (EDA) to analyse gym data, then to model the calories burnt at gym.
The project contains 2 main parts:
* Exploratory Data Analysis
* Modelling the calories burnt


## Dataset
We use the gym dataset that we downloadeded from Kaggle (https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset).
It contains 973 rows and 15 variables, the following ones: 
* age: age of the gym member
* gender: gender of the gym member (Male or Female)
* weight: weight of the gym member (in kilograms)
* height: height of the gym member (in meters)
* max_bpm: maximum heart rate (beats per minute) of the gym member during the workout session
* avg_bpm: average heart rate (beats per minute) of the gym member during the workout session
* resting_bpm: heart rate (beats per minute) of the gym member before the workout session
* session_duration: duration of the workout session in hours
* calories_burnt: calories burnt by the gym member during the workout session
* workout_type: type of workout performed during the session by the gym member (Cardio, HIIT, Strength or Yoga)
* fat_percentage: body fat percentage of the gym member
* water_intake: water intake during the workout by the gym member
* workout_frequency: number of workout sessions per week
* experience_level: level of experience of the gym member (1 for beginner, 2 for medium and 3 for expert)
* bmi: Body Mass Index computed as the weight divided by the squared height 


## Exploratory Data Analysis

### Univariate data analysis
The univariate data analysis aims at investigating each variable 1 by 1. During this step, we can check for missing values and outliers for each variable. This step also allows to better understand the data, checking each variable and looking at their distribution (skewness or symmetry).
Insights we get from the univariate data analysis:
* gender: 
    * this variable doesn't contain any missing values
    * there is no outliers: we only have 2 different values 'Female' and 'Male' 
    * we notice that the proportion of men vs women is pretty similar (47.48% of women vs 52.52% of men)
* age: 
    * this variable doesn't contain any missing values
    * we don't see any outliers either
    * we notice that more than 50% are between 28 and 49yo (we were rather expecting people to be mostly between 25 and 35yo). The variable distribution looks like uniform law.
* BPM (resting BPM, average BPM and max BPM): 
    * these variables don't contain any missing values
    * we don't see any outliers either
    * we notice (as expected) that resting bpm < average bpm < max bpm. Moreover, we notice that the spread of average bpm and max bpm are bigger than the spread of resting bpm, meaning that the bpms of people are less different when people are rested than when they do sport. It also means that bpms of people are more different when they do sport (probably because level of sport is different based on people and in general people spend more time resting than exercising)
    * the density distributions are pretty symmetric, almost uniform law
* weight: 
    * this variable doesn't contain any missing values
    * we see a couple of values that could be considered as outliers at the top of the boxplot (but not that extreme)
    * we notice that 50% of the gym members (from this dataset) have a weight between 58.1 and 86kg. Moreover, the distribution is not so symmetric, we observe a positive/right skewness
* height: 
    * this variable doesn't contain any missing values
    * we don't see any outliers either
    * we notice that 50% of the gym members (in this dataset) have a height between 1.62 and 1.8m. Moreover, the distribution is pretty much symmetric, eventually slightly right/positively skewed
* session_duration: 
    * this variable doesn't contain any missing values
    * we don't see any outliers either
    * we notice that more than 50% of the gym members spend between 1h and 1.5h at the gym. We observe a very symmetric distribution 
* calories_burnt: 
    * this variable doesn't contain any missing values
    * we see a couple of values that could be considered as outliers at the top (but not that extreme)
    * we notice that more than 50% of the gym members are burning between 720 and 1080 calories. Moreover we observe a distribution rather symmetric
* workout_type: 
    * this variable doesn't contain any missing values
    * there is no outliers: we only have 4 different values ('Cardio', 'HIIT', 'Strength' and 'Yoga')
    * we notice that the frequency between the 4 sports is similar: HIIT is the less represented (22.71%) and strength is the most represented (26.52%)
* fat_percentage: 
    * this variable doesn't contain any missing values
    * we don't see any outliers either
    * we notice that more than 50% of the gym members in this dataset have a fat percentage between 21% and 30%. Moreover, we observe that the distribution is not so symmetric, it's rather negatively/left skewed
* water_intake: 
    * this variable doesn't contain any missing values
    * we don't see any outliers either
    * we notice that more than 50% of the gym members (in this dataset) are drinking between 2.2 and 3.1 liters of water. Moreover, we observe that the distribution is rather symmetric, except a big down on 3 (bimodal distribution)
* workout frequency: 
    * this variable doesn't contain any missing values
    * there is no outliers: we only have 4 different values (from 2 to 5)
    * we notice that nearly 70% of the gym members (in this dataset) go between 3 and 4 times to the gym
* experience level: 
    * this variable doesn't contain any missing values
    * there is no outliers: we only have 3 different values (from 1 to 3 for beginner to expert)
    * we notice that most of the gym members (in this dataset) are medium experienced (41.73%), beginners represent 38.64% of gym members and experts represent 19.63%
* BMI: 
    * this variable doesn't contain any missing values
    * we see a couple of values that could be considered as outliers on the top (but not that extreme)
    * we notice that more than 50% have a BMI between 20.1 and 28.6. The health litterature says that a BMI under 18.5 means that the person is very underweight, a BMI between 18.5 and 24.9 means that the person has a healthy weight, a BMI between 25 and 29.9 means that the person is overweight and a BMI higher than 30 means that the person is obese. It's good to remember that the BMI is computed based on the weight but doesnt differentiate between body fat and muscle mass so there are some exceptions to the BMI guidelines. Based on the data we have, we find that 17.27% of our dataset is underweight, 38.03% of our dataset has an healthy weight, 24.97% of our dataset is overweight and 19.73% of our dataset is obese, so this dataset is pretty varied. We observe that the distribution is rather positively skewed.

The univariate data analysis helps us to conclude that:
* the data has no missing value
* the data has a couple of outliers but since they are not that extreme and the dataset is not that big, we will just keep them. Some values are a bit higher than the rest of the values but we don't see such a difference that would make the model been shifted.


### Bivariate data analysis
We are now doing the bivariate data analysis (analyse each variable with respect to the variable we will want to predict: calories_burnt). Burnt calories variable is numerical and we have both categorical and numerical feature variables so we will carry out 2 types of associations/analysis:
* numerical-numerical relationship: for each numerical feature variable, we will draw the scatterplot between this variable and the calories_burnt variable. We will also compute the correlation coefficient between this variable and the calories burnt variable (with Kendall method because Pearson only catches the linear relationship). This is what we notice about the relationship between calories burnt and each of the following variables (based on the scatterplot and the correlation):
    * age: no strong association (cor = -0.1, negative correlation means that the older the person is, the less calories are burnt ; indeed the body metabolism is less active the older we get [but it might also comes from other variables])
    * weight: no strong association (cor = 0.09)
    * height: no strong association (cor = 0.05)
    * max_bpm: no strong association (cor = -0.01)
    * avg_bpm: positive association (cor = 0.23, positive correlation, meaning that the higher is the BPM on average during the session, the more calories are burnt [this is well known, fast heart rate makes the calories being burnt])
    * resting_bpm: no strong association (cor = -0.0)
    * session_duration: strong positive association (cor = 0.74, strong positive correlation means that the longer the workout session is, the more calories are burnt)
    * fat_percentage: strong negative association (cor = -0.35)
    * water_intake: positive association (cor = 0.24, positive correlation means that the more water the member has drunk during the session, the more calories is burnt)
    * bmi: no strong association (cor = 0.05)
* numerical-categorical relationship: for each categorical variable, we:
    * draw the boxplot of the calories burnt variable (Y) for each class of the categorical variable (X): to check the distribution of the calories burnt for each class
    * draw the density plot of the calories burnt for each class of the categorical variable: to check if the density is somewhat normal (the sample is pretty large anyway)
    * carry on the Levene test: to check if the variance of Y variable is the same among the class of X variable
    H0: variances of calories_burnt on all X groups are the same
    H1: variances of calories_burnt on all X groups are different
    If pvalue < 0.05, we reject H0: variances of Y variable among X categories are different
    * carry on the T-test: to test for significant link between X and Y (if X has only 2 classes), it determines if there's a significant difference between the means of 2 groups
    H0: the true means of Y among different classes of X are equal
    H1: the true mean of Y among the different groups of X is different
    If p-value < 0.05, we reject H0, meaning that the true means of Y is significantly different for the different groups of X
    * carry on ANOVA test: to test for significant link between X and Y (if X has more than 2 classes and the variance of Y is the same among classes of X). It tests if the mean of Y over X categories is the same. Assumptions: normality of data (checked by density plot), independence of observations (for this project, it's always the case, records can't be in both groups), homogeneity of variance (we checked it with Levene test, otherwise we carry on a Kruskal-Wallis test)
    H0: the mean of variable Y across the categories of variable X are equal (no difference in means)
    H1: at least one group mean significantly differs
    If pvalue < 0.05, we reject H0 and conclude that at least one group has a significantly different mean and we conclude that variable X significantly affects variable Y
    * carry on Kruskal-Wallis test: to test for significant link between X and Y (if X has more than 2 classes and variance of Y is not the same among classes of X)
    H0: means of Y among X categories are the same
    H1: means of Y among X categories are different
    If pvalue < 0.05, we reject H0: mean of calories burnt is different depending on the groups of X.

    This is what we notice about the relationship between calories burnt and each of the following categorical variables (based on the boxplot, density plot and the different tests):
    * gender: from the boxplot, we notice that male burn more calories than woman (but it can be linked to other variables, for example men are training longer or harder, or are more expert). 
    From the density plot, we notice that the distribution of Y is normal for both classes. We assume normality since we have a pretty big dataset and the density plot shows us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue < 0.05 so we reject H0: variance of Y is different for both gender groups.
    We carry on a 2 samples T-test (since gender has only 2 classes: Male and Female) with unequal variances. T-test pvalue < 0.05 so we reject H0, the true means of calories_burnt among female and male are different, so calories_burnt and gender variables are significantly linked.
    * workout_type: from the boxplot, we don't see so much difference in the calories_burnt depending on the sport. 
    From the density plot, we notice that the distribution of Y is normal for all X classes. We assume normality since we have a pretty big dataset and the density plot shows us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue > 0.05 so we don't reject H0: variance of Y is the same for all workout type groups.
    We carry on an ANOVA test (since workout type has more than 2 classes: 'Cardio', 'HIIT', 'Strength' and 'Yoga') with equal variances. ANOVA pvalue > 0.05 so we don't reject H0, mean of calories burnt is the same for the 4 different workout types, so calories_burnt and workout_type variables are not significantly linked.
    * experience_level: from the boxplot, we do see difference in the calories_burnt depending on the experience level. The more experienced the member is, the more calories the member will burn. It's pretty straightforward: expert burn more, probably because they train harder or know what to train to burn calories.
    From the density plot, we notice that the distribution of Y is normal for all classes. We assume normality since we have a pretty big dataset and the density plot shows us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue < 0.05 so we reject H0: variance of Y are not the same for all groups of experience_level, so we can't use ANOVA test anymore.
    We carry on a Kruskal-Wallis test (since experience level has more than 2 classes: 1, 2 and 3) with unequal variances. Kruskal-Wallis pvalue < 0.05 so we reject H0: mean of calories burnt is different depending on the experience level, so calories_burnt and experience_level variables are significantly linked.
    * workout_frequency: from the boxplot, we do see difference in the calories_burnt depending on the workout_frequency. The more often the member goes to the gym, the more calories the member will burn. It's pretty straightforward: people who come often to the gym burn more, probably because they are more expert.
    From the density plot, we notice that the distribution of Y is normal for all classes. We assume normality since we have a pretty big dataset and the density plot shows us pretty normal data.
    We carry on Levene test to check if the variances are equal. Levene pvalue < 0.05 so we reject H0: variance of Y are not the same for all groups of workout_frequency, so we can't use ANOVA test anymore.
    We carry on a Kruskal-Wallis test (since workout_frequency has more than 2 classes:  2, 3, 4 and 5) with unequal variances. Kruskal-Wallis pvalue < 0.05 so we reject H0: mean of calories burnt is different depending on the workout_frequency, so calories_burnt and workout_frequency variables are significantly linked.

To sum up, we find the stronger relationship between calories_burnt variable and the following (X) variables: avg_bpm, session_duration, fat_percentage, water_intake, gender, experience_level and workout_frequency.


## Modelling the calories burnt
We want to predict the calories burnt at the gym using all other variables. We use several models and we compare which model gives the best results.

### Data processing
Before modelling we do a couple of data processing:
* transform all variables to numeric:
    * for ordinal variables (experience level and workout frequency): we just switch the type to integer
    * for nominal variables (gender and workout type): we encode the variables using LabelEncoder for gender (from Female/Male to 0/1) and using OneHotEncoder for workout type (from 'Cardio' to [1 0 0], from 'HIIT' to [0 1 0], from 'Strength' to [0 0 1] and from 'Yoga' to [0 0 0])
* drop BMI variable because it's built from weight and height variables so keeping it would introduce multi-colinearity. Avoiding multicollinearity is important for building robust, interpretable, and generalizable machine learning models.
* split X (input features) and y (target labels)
* split the dataset into train and test sets: we keep 20% for the test set
* normalize the train set then the test set to avoid having the model being mainly influenced by a set of variables only because they have a bigger scale (some models are very sensible to this, like KNN). We normalize the train set before the test set to be the closest from what would happen with a real-world application: the model would encounter unseen data (test set) and would need to make predictions based on patterns learned from the training data alone. By doing this we make sure that the model is trained only on information from the training data, we apply the same normalization to the test set to ensure consistent scaling, but do not fit the scaler to the test set.


### Models
We use several models to model y with respect to X and choose the best one. 
To train the models, we use cross-validation, it allows to split the training set in smaller samples to have more training samples and the model is trained several times (for each CV sample).
For each model, we set up an hyperparameters grid search, the model will train for each combination of parameters and determine the best parameters for this model. The number of possible candidates is determined by multiplying the number of options for each parameter (and finally multiply by the number of folders from the cross-validation).

Following are the models we tried for this project:
* Decision trees: the algorithm splits the data into subsets based on feature values, creating a tree-like structure of decisions (made with root, nodes, branches and leaves). At the beginning, the root node contains the whole dataset. Then at every step (at each decision node), the tree splits the data from the node based on a condition on a feature to maximize the purity of the subsets. The leaves contain the predicted output (mean in case of a regression). It is a simple, interpretable machine learning algorithm but it is sensitive to small changes in data and it is very easy to overfit (especially with deep trees). To make a prediction, the tree follows the splits from the root to a leaf node. We use the following parameters to train a decision tree model:
    * max_depth: the maximum depth of the tree (limiting the depth can help prevent overfitting by controlling the complexity of the model)
    * min_samples_split: the minimum number of samples required to split an internal node (larger values prevent the model from creating nodes that are too specific to the training data)
    * min_samples_leaf: the minimum number of samples required to be at a leaf node (increasing this value can smooth the model, making it more generalized)
    * max_features: the number of features to consider when looking for the best split. Several options are possible: all features (None), the square root of the number of features ('sqrt'), an integer
    * splitter: the strategy used to split at each node. Several options are possible: 'best' (to choose the best split) or 'random' (to choose a random split)
    * max_leaf_nodes: the maximum number of leaf nodes in the tree (limiting this helps prevent overfitting by restricting the complexity of the tree)
* K-Nearest Neighbor: the algorithm makes predictions based on the k closest data points (neighbors) in the training set. For regression, the predicted value is the average (or weighted average) of the k nearest neighbors' target values. KNN uses a distance metric (commonly Euclidean distance) to find the closest neighbors. It's a simple, easy to understand and effective (with small and medium datasets) algorithm but it's computationally expensive (especially for large datasets). We use the following parameters to train a KNN model:
    * n_neighbors: the number of neighbors to use when making predictions (lower values may lead to more variance, and higher values may lead to more bias)
    * weights: defines how the distance to neighbors influences the prediction ('uniform' gives all neighbors equal weight, while 'distance' gives closer neighbors more weight)
    * algorithm: the algorithm to compute the nearest neighbors ('auto' chooses the best algorithm based on the data)
    * metric: the distance metric to use for finding neighbors (common options include 'euclidean' and 'manhattan')
* Random Forest: the algorithm is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance. To make a prediction (in case of regression like here) it averages the predictions from all the trees. Each tree is built using bootstrapped samples (random subsets with replacement) from the training data, and at each split, a random subset of features is considered to prevent overfitting. This algorithm reduces overfitting compared to individual decision trees, it handles large datasets well and it can capture complex relationships in the data but it's more computationally expensive and less interpretable than a single decision tree. We use the following parameters to train a random forest model:
    * n_estimators: the number of trees in the forest (a higher number generally leads to a better model, but it also increases computational cost)
    * max_depth: the maximum depth of the tree (larger depths can capture more complex relationships but may also overfit)
    * min_samples_split: the minimum number of samples required to split an internal node.
    * min_samples_leaf: the minimum number of samples required to be at a leaf node.
    * max_features: the number of features to consider when looking for the best split (common options include 'auto' (all features), 'sqrt' (square root of the number of features) or an integer)
    * bootstrap: whether to use bootstrap samples when building trees (set to True to sample with replacement)
* Gradient Boosting: this algorithm is an ensemble learning technique. It builds a model by combining multiple weak learners in a sequential manner (one at a time), where each tree tries to correct the errors (residuals) of the previous one. The "gradient" in Gradient Boosting refers to using gradient descent to minimize the loss (error) function by adjusting the weights of the trees. This model often achieves high accuracy and is robust to overfitting with proper tuning but it's computationally expensive and slower to train compared to other models, it's sensitive to noisy data and outliers and it requires careful tuning of hyperparameters (e.g., learning rate, number of estimators). We use the following parameters to train a gradient boosting model:
    * n_estimators: the number of boosting stages (trees) to fit (more trees can lead to a more complex model)
    * learning_rate: the step size to shrink the contribution of each tree (smaller values may lead to better performance but require more trees)
    * subsample: the fraction of samples used for fitting each tree (lower values help in reducing overfitting)
    * max_depth: the maximum depth of the individual decision trees (deeper trees can model more complex relationships)
    * min_samples_split: the minimum number of samples required to split an internal node (higher values prevent the model from learning overly specific patterns)
    * min_samples_leaf: the minimum number of samples required to be at a leaf node (this helps in smoothing the model)
    * max_features: the number of features to consider when looking for the best split (to be tune to prevent overfitting and speed up the model)
* Extreme Gradient Boosting (XGBoost): the algorithm is an optimized, highly efficient and fast version of the gradient boosting model. It's designed to perform better in terms of speed, accuracy, and scalability. XGBoost builds trees sequentially, where each tree corrects the errors of the previous one. XGBoost includes L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting, making it more robust compared to traditional gradient boosting. XGBoost supports parallel processing during training, making it faster than standard gradient boosting algorithms. It can be hard to interpret (like most tree-based models) compared to simpler models. We use the following parameters to train a XGBoost model:
    * n_estimators: the number of boosting rounds (trees) (more trees generally lead to a better model, but too many can overfit)
    * learning_rate: determines how much each tree contributes to the final model (smaller values are typically preferred as they slow down the learning process, which may improve performance but requires more trees)
    * max_depth: controls the depth of each tree (larger depths can capture more complex relationships but may also overfit)
    * min_child_weight: controls overfitting by determining the minimum sum of instance weights (of the gradients) in a child (larger values prevent the model from learning overly specific patterns)
    * subsample: fraction of the data to use for training each tree (by setting it under 1, some randomness is introduced into the model and help reduce overfitting)
    * colsample_bytree: fraction of features to use for each tree (by setting it under 1, some randomness is introduced into the model and help reduce overfitting)
    * gamma: controls the regularization of the tree, it defines the minimum reduction in the loss required to make a further split in the tree (higher values make the algorithm more conservative by preventing further splits)
    * reg_alpha: L1 (Lasso) regularization term on weights (it helps in controlling the overfitting). This parameter adds a penalty term to the loss function that is proportional to the absolute value of the model's coefficients. It tends to shrink the weights of irrelevant features to zero. This can help reduce the models complexity by effectively eliminating some features from the final model.
    * reg_lambda: L2 (Ridge) regularization term on weights (it helps in controlling the overfitting). This parameter adds a penalty term to the loss function that is proportional to the square of the model's coefficients. It helps in shrinking the coefficients of features but does not shrink them to zero. Instead of eliminating features, L2 regularization reduces the magnitude of their weights, making the model less sensitive to small fluctuations in the training data and reducing overfitting.
    * objective: loss function ('reg:squarederror' is the most common for regression task)
    * booster: type of boosting model. Several options are possible: 'gbtree' (tree-based), 'gblinear' (linear so rather faster but may not perform so well on non-linear problems), or 'dart' (Dropout trees, a regularization technique that can help with overfitting)
* Support Vector Regressor: it aims at finding the best hyperplane that approximates the data while maintaining a margin of tolerance for errors (up to a certain threshold epsilon). Data points within this margin are considered as correct. SVR can use different kernel functions (linear, polynomial, radial basis function, etc.) to map data into higher-dimensional spaces, allowing it to capture non-linear relationships. Only a subset of the data points, called support vectors, influence the model. These points are the ones closest to the margin and play a key role in defining the model. SVR is an effective model for both linear and non-linear regression tasks, it's robust to overfitting, especially with appropriate kernel selection and regularization but it can be computationally expensive, especially for large datasets and the choice of kernel, regularization, and epsilon values can be tricky to tune. We use the following parameters to train a SVR model:
    * kernel: specifies the type of kernel to use in the SVR. Several options are possible: 'linear' (for linear regression), 'poly' (for polynomial kernel), 'rbf' (Radial Basis Function, for non-linear regression, [most common])
    * C: the regularization parameter (penalty for errors). It trades off the model's ability to fit the training data versus its ability to generalize to new data (a higher C value means the model tries to fit the training data more closely [but might overfit], while a lower C value allows more error but improves generalization)
    * epsilon: the margin of tolerance within which no penalty is given for errors (a small epsilon value makes the model sensitive to the training data, while a larger value allows more margin for error)
    * degree: the degree of the polynomial kernel function (only used when the kernel is 'poly') (higher degrees can model more complex relationships, but may also increase the risk of overfitting)
    * gamma: the kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels, it determines the influence of a single training example on the decision boundary. Several options are possible: 'scale' which is 1 / (n_features * variance of X) (generally a good default), 'auto' which is 1 / n_features, a number (a small value makes the model more general, while a larger value makes it more sensitive to individual points and increase complexity and risk of overfitting)
    * tol: the tolerance for stopping criteria (smaller values will make the algorithm run for longer but may result in more accurate results)
    * cache_size: the size of the kernel cache in MB. A larger cache can speed up computation for large datasets but may use more memory
    * shrinking: whether to use the shrinking heuristic to speed up training. When True, it allows the SVR algorithm to ignore support vectors that do not significantly affect the optimization process. Essentially, it uses an approximation to improve the efficiency of the model training.

We compare the MAE, MSE and RMSE of all models to select the best model.
