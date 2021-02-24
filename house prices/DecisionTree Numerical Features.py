# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from sklearn.tree import DecisionTreeClassifier
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a
# version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test.head()

train.describe()
train['SalePrice'].describe()
plt.hist(train['SalePrice'])
test.describe()
train.info()
test.info()

train.isnull().sum().sort_values(ascending=False).head(20)
test.isnull().sum().sort_values(ascending=False).head(20)

# PoolQC (object), MiscFeature (object), Alley(object) y Fence(object) have too many null values in both datasets.
# Therefore, we eliminate them. They do not provide information

train_edited = train.drop(['PoolQC'], axis=1)
train_edited = train_edited.drop(['MiscFeature'], axis=1)
train_edited = train_edited.drop(['Alley'], axis=1)
train_edited = train_edited.drop(['Fence'], axis=1)
train_edited = train_edited.drop(['Id'], axis=1)

test_edited = test.drop(['PoolQC'], axis=1)
test_edited = test_edited.drop(['MiscFeature'], axis=1)
test_edited = test_edited.drop(['Alley'], axis=1)
test_edited = test_edited.drop(['Fence'], axis=1)
test_edited = test_edited.drop(['Id'], axis=1)

# Now, we have to compare the features with the final tag.
# First of all, we're going to compare the numerical fields

train_edited_numeric = train_edited.select_dtypes(np.number)

train_edited_numeric.isnull().sum().sort_values(ascending=False).head(3)

# We calculate the averages for each field and then replace the null values by these values

average_LotFrontage = train_edited_numeric['LotFrontage'].median()
average_GarageYrBlt = train_edited_numeric['GarageYrBlt'].median()
average_MasVnrArea = train_edited_numeric['MasVnrArea'].median()

train_edited_numeric['LotFrontage'].fillna(average_LotFrontage, inplace=True)
train_edited_numeric['GarageYrBlt'].fillna(average_GarageYrBlt, inplace=True)
train_edited_numeric['MasVnrArea'].fillna(average_MasVnrArea, inplace=True)

train_edited_numeric.isnull().sum().sort_values(ascending=False).head(3)

# As we can see, the numerical variables no longer have null values.
# We now proceed to plot each of the above variables respect to the final price using scatter diagrams in order to see
# any relationship between them. We also use simple linear regression to see the trend of the data

sale_price = train_edited_numeric['SalePrice']

# We need to rename a couple of fields so that the linear regression does not give syntax problems
train_edited_numeric = train_edited_numeric.rename(columns={'1stFlrSF': 'FirstFlrSF',
                                                            '2ndFlrSF': 'SecondFlrSF',
                                                            '3SsnPorch': 'ThirdSsnPorch'})
numeric_columns = train_edited_numeric.columns
numeric_columns = numeric_columns[:36]

fig = plt.figure()
nrows = 9
ncols = 4
order = 0
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(20, 35))
for row in range(0, nrows):
    for column in range(0, ncols):
        formula = 'SalePrice' + '~' + str(numeric_columns[order])
        axes[row, column].scatter(train_edited_numeric[numeric_columns[order]], sale_price)
        axes[row, column].set_xlabel(numeric_columns[order], labelpad=5)
        lm = smf.ols(formula=formula, data=train_edited_numeric).fit()
        expected_price = lm.predict(pd.DataFrame(train_edited_numeric[numeric_columns[order]]))
        axes[row, column].plot(pd.DataFrame(train_edited_numeric[numeric_columns[order]]), expected_price, c="red",
                               linewidth=2)
        order = order + 1

# From the above variables we can see a certain relationship with:
# OverallQual TotalBsmtSF 1stFlrSF 2ndFlrSF GrLivArea GarageArea
# Now we have to repeat the procedure for the test set
test_edited_numeric = test_edited.select_dtypes(np.number)
test_edited_numeric.isnull().sum().sort_values(ascending=False).head(11)

average_LotFrontage_test = test_edited_numeric['LotFrontage'].median()
average_GarageYrBlt_test = test_edited_numeric['GarageYrBlt'].median()
average_MasVnrArea_test = test_edited_numeric['MasVnrArea'].median()
average_BsmtHalfBath_test = test_edited_numeric['BsmtHalfBath'].median()
average_BsmtFullBath_test = test_edited_numeric['BsmtFullBath'].median()
average_TotalBsmtSF_test = test_edited_numeric['TotalBsmtSF'].median()
average_GarageCars_test = test_edited_numeric['GarageCars'].median()
average_BsmtFinSF1_test = test_edited_numeric['BsmtFinSF1'].median()
average_BsmtFinSF2_test = test_edited_numeric['BsmtFinSF2'].median()
average_BsmtUnfSF_test = test_edited_numeric['BsmtUnfSF'].median()
average_GarageArea_test = test_edited_numeric['GarageArea'].median()

test_edited_numeric['LotFrontage'].fillna(average_LotFrontage_test, inplace=True)
test_edited_numeric['GarageYrBlt'].fillna(average_GarageYrBlt_test, inplace=True)
test_edited_numeric['MasVnrArea'].fillna(average_MasVnrArea_test, inplace=True)
test_edited_numeric['BsmtHalfBath'].fillna(average_BsmtHalfBath_test, inplace=True)
test_edited_numeric['BsmtFullBath'].fillna(average_BsmtFullBath_test, inplace=True)
test_edited_numeric['TotalBsmtSF'].fillna(average_TotalBsmtSF_test, inplace=True)
test_edited_numeric['GarageCars'].fillna(average_GarageCars_test, inplace=True)
test_edited_numeric['BsmtFinSF1'].fillna(average_BsmtFinSF1_test, inplace=True)
test_edited_numeric['BsmtFinSF2'].fillna(average_BsmtFinSF2_test, inplace=True)
test_edited_numeric['BsmtUnfSF'].fillna(average_BsmtUnfSF_test, inplace=True)
test_edited_numeric['GarageArea'].fillna(average_GarageArea_test, inplace=True)

test_edited_numeric.isnull().sum().sort_values(ascending=False).head(11)

test_edited_numeric = test_edited_numeric.rename(columns={'1stFlrSF': 'FirstFlrSF',
                                                          '2ndFlrSF': 'SecondFlrSF'})

# Let's make a first approximation with these variables to see how well or poorly our decision tree
# calculates the final price.


# --------- Classifier ---------
feature_cols = ['OverallQual', 'TotalBsmtSF', 'FirstFlrSF', 'GrLivArea', 'GarageArea']

# X = Features
X = train_edited_numeric[feature_cols]
# y = target variable
y = train_edited_numeric.SalePrice

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X, y)

# Predict the response for test dataset
y_pred = clf.predict(test_edited_numeric[feature_cols])

output = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# 0.24750