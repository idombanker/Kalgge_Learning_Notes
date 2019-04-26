# Feature Engineering
Basically, our approach is to make as many features as possible and then give them all to the model to use! 
Later, we can perform **feature reduction** using the feature importances from the model or other techniques such as PCA.

### Usefulness 
To determine if the new variable is useful, we can calculate the **Pearson Correlation Coefficient** (r-value) between this variable and the target.
This measures the strength of a linear relationship between two variables and ranges from -1 (perfectly negatively linear) to +1 (perfectly positively linear). 

The r-value is **not best measure** of the "usefulness" of a new variable, but it can give a first approximation of whether a variable will be helpful to a machine learning model. 

Therefore, we look for the variables with the greatest **absolute** value r-value relative to the target.

We can also visually inspect a relationship with the target using the **Kernel Density Estimate**(KDE) plot.

### new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
new_corrs 里面的元素是元祖，所以要设置排序的ｋｅｙ．这里的ｋｅｙ是元祖的第一个元素．

### The Multiple Comparisons Problem
 We can make hundreds of features, and some will turn out to be corelated with the target simply because of random noise in the data. Then, when our model trains, it may overfit to these variables because it thinks they have a relationship with the target in the training set, but this does not necessarily generalize to the test set. 
 这也许就是为什么我之前加入一个和频率相关的数据之后，验证集的分数反而下降了．
 
### 接下来用ＰＣＡ把数据搞一搞
 
### Feature Selection
We can also use the percentage of missing values to remove features with a substantial majority of values that are not present.

Feature selection will be an important focus going forward, because reducing the number of features can help the model learn during training and also generalize better to the testing data.

Feature selection is the process of removing variables to help our model to learn and **generalize better to the testing set**.

There are a number of tools we can use for this process, but in this notebook we will stick to removing columns with a high percentage of missing values and variables that have a high correlation with one another. Later we can look at using the feature importances returned from models such as the **Gradient Boosting Machine or Random Forest to perform feature selection.**

### 用shift window的方法，去创造，更多的样本．
#### Correlated 不等于有用
 However, just because the variable is correlated does not mean that it will be useful, and we have to remember that if we generate hundreds of new variables, some are going to be correlated with the target simply because of **random noise.**
 
 ### 去除相关性强的数据也许会导致分数下降
 Removing the highly collinear variables slightly decreases performance so we will want to consider a different method for feature selection. 
 Moreover, we can say that some of the features we built are among the most important as judged by the model.
### Automated Feature Engineering
Feature Tools,
**an open-source Python library for automatically creating features with relational data.**
Feature tools has the ability to do this for us, creating many new candidate features with minimal effort. These features are combined into a single table that can then be passed on to our model.

### 保持乐观，韧性，并且努力加油！

### Feature Tools
 The concept of Deep Feature Synthesis is to use basic building blocks known as feature primitives (like the transformations and aggregations done above) that can be stacked on top of each other to form new features. 
The depth of a "deep feature" is equal to the number of **stacked primitives**.

### deep feature synthesis
就是堆叠在一起，同我的wavelet深度堆积
### https://blog.csdn.net/Mlooker/article/details/80318601　



