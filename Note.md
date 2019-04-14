## Feature Engineering
Basically, our approach is to make as many features as possible and then give them all to the model to use! 
Later, we can perform **feature reduction** using the feature importances from the model or other techniques such as PCA.

# Usefulness 
To determine if the new variable is useful, we can calculate the **Pearson Correlation Coefficient** (r-value) between this variable and the target.
This measures the strength of a linear relationship between two variables and ranges from -1 (perfectly negatively linear) to +1 (perfectly positively linear). 

The r-value is **not best measure** of the "usefulness" of a new variable, but it can give a first approximation of whether a variable will be helpful to a machine learning model. 

Therefore, we look for the variables with the greatest **absolute** value r-value relative to the target.

We can also visually inspect a relationship with the target using the **Kernel Density Estimate**(KDE) plot.

# new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
new_corrs 里面的元素是元祖，所以要设置排序的ｋｅｙ．这里的ｋｅｙ是元祖的第一个元素．

# The Multiple Comparisons Problem
 We can make hundreds of features, and some will turn out to be corelated with the target simply because of random noise in the data. Then, when our model trains, it may overfit to these variables because it thinks they have a relationship with the target in the training set, but this does not necessarily generalize to the test set. 
