# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import statistics as stat
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output
# when you create a version using "Save & Run All"
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
train_edited = train_edited.drop(['FireplaceQu'], axis=1)
train_edited = train_edited.drop(['Id'], axis=1)

test_edited = test.drop(['PoolQC'], axis=1)
test_edited = test_edited.drop(['MiscFeature'], axis=1)
test_edited = test_edited.drop(['Alley'], axis=1)
test_edited = test_edited.drop(['Fence'], axis=1)
test_edited = test_edited.drop(['FireplaceQu'], axis=1)
test_edited = test_edited.drop(['Id'], axis=1)

# Now, we have to comparte the features with the final tag.
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
# We now proceed to plot each of the above variables respect to the final price using scatter diagrams
# in order to see any relationship between them. We also use simple linear regression to see the trend of the data.

sale_price = train_edited_numeric['SalePrice']

# We need to rename a couple of fields so that the linear regression does not give syntax problems
train_edited_numeric = train_edited_numeric.rename(columns={'1stFlrSF':'FirstFlrSF',
                                                           '2ndFlrSF':'SecondFlrSF',
                                                           '3SsnPorch':'ThirdSsnPorch'})
numeric_columns = train_edited_numeric.columns
numeric_columns = numeric_columns[:36]

fig = plt.figure()
nrows=9
ncols=4
order = 0
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(20,35))
for row in range(0, nrows):
    for column in range(0, ncols):
        formula = 'SalePrice'+'~'+str(numeric_columns[order])
        axes[row, column].scatter(train_edited_numeric[numeric_columns[order]], sale_price)
        axes[row, column].set_xlabel(numeric_columns[order], labelpad = 5)
        lm = smf.ols(formula=formula, data=train_edited_numeric).fit()
        expected_price = lm.predict(pd.DataFrame(train_edited_numeric[numeric_columns[order]]))
        axes[row, column].plot(pd.DataFrame(train_edited_numeric[numeric_columns[order]]), expected_price, c="red", linewidth=2)
        order = order + 1


# From the above variables we can see a certain relationship with:
# OverallQual TotalBsmtSF 1stFlrSF 2ndFlrSF GrLivArea GarageArea
# To be more certain, we calculate the correlation matrix for the variables

plt.figure(figsize=(20, 20))
correlation_mat = train_edited_numeric.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()

corr_pairs = correlation_mat.unstack()
print(corr_pairs['SalePrice'])

strong_pairs = corr_pairs['SalePrice'][abs(corr_pairs['SalePrice']) > 0.5]
print(strong_pairs.sort_values(ascending=False))
# The following characteristics have the highest correlation with the final selling price
# OverallQual 0.790982  GrLivArea 0.708624  GarageCars 0.640409
# GarageArea 0.623431  TotalBsmtSF 0.613581  FirstFlrSF 0.605852

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

test_edited_numeric = test_edited_numeric.rename(columns={'1stFlrSF':'FirstFlrSF',
                                                           '2ndFlrSF':'SecondFlrSF'})

# Now it is time to analyze the categorical variables
train_edited_object = train_edited.select_dtypes(np.object)

# We check if the dataframe has null values in order to replace them.
train_edited_object.isnull().sum().sort_values(ascending=False).head(11)

# Since these are categorical variables we cannot calculate the median as we have done before,
# we now choose to use the most repeated value.

mode_GarageCond = stat.mode(train_edited_object['GarageCond'])
mode_GarageQual = stat.mode(train_edited_object['GarageQual'])
mode_GarageFinish = stat.mode(train_edited_object['GarageFinish'])
mode_GarageType = stat.mode(train_edited_object['GarageType'])
mode_BsmtExposure = stat.mode(train_edited_object['BsmtExposure'])
mode_BsmtFinType2 = stat.mode(train_edited_object['BsmtFinType2'])
mode_BsmtCond = stat.mode(train_edited_object['BsmtCond'])
mode_BsmtFinType1 = stat.mode(train_edited_object['BsmtFinType1'])
mode_BsmtQual = stat.mode(train_edited_object['BsmtQual'])
mode_MasVnrType = stat.mode(train_edited_object['MasVnrType'])
mode_Electrical = stat.mode(train_edited_object['Electrical'])

train_edited_object['GarageCond'].fillna(mode_GarageCond, inplace=True)
train_edited_object['GarageQual'].fillna(mode_GarageQual, inplace=True)
train_edited_object['GarageFinish'].fillna(mode_GarageFinish, inplace=True)
train_edited_object['GarageType'].fillna(mode_GarageType, inplace=True)
train_edited_object['BsmtExposure'].fillna(mode_BsmtExposure, inplace=True)
train_edited_object['BsmtFinType2'].fillna(mode_BsmtFinType2, inplace=True)
train_edited_object['BsmtCond'].fillna(mode_BsmtCond, inplace=True)
train_edited_object['BsmtFinType1'].fillna(mode_BsmtFinType1, inplace=True)
train_edited_object['BsmtQual'].fillna(mode_BsmtQual, inplace=True)
train_edited_object['MasVnrType'].fillna(mode_MasVnrType, inplace=True)
train_edited_object['Electrical'].fillna(mode_Electrical, inplace=True)

train_edited_object['SalePrice'] = train['SalePrice']
object_columns = train_edited_object.columns
object_columns = object_columns[:38]

for feature in object_columns:
    train_edited_object_group = train_edited_object.groupby([feature]).median()
    train_edited_object_group.plot.bar()

for feature in object_columns:
    train_edited_object_group = train_edited_object.groupby([feature]).mean()
    train_edited_object_group.plot.bar()

# It appears that there may be a clear difference in the selling price depending on the following features:
# - Street (2) - Utilities (2) - MasVnrType (4) - ExterQual (4) - ExterCond (5) - BsmtQual (4)
# - BmstCond (4) - CentralAir (2) - Electrical (5) - KitchenQual (4) - GarageFinish (3) - PavedDrive (3)

test_edited_object = test_edited.select_dtypes(np.object)

test_edited_object.isnull().sum().sort_values(ascending=False).head(17)

mode_GarageCond_test = stat.mode(test_edited_object['GarageCond'])
mode_GarageQual_test = stat.mode(test_edited_object['GarageQual'])
mode_GarageFinish_test = stat.mode(test_edited_object['GarageFinish'])
mode_GarageType_test = stat.mode(test_edited_object['GarageType'])
mode_BsmtCond_test = stat.mode(test_edited_object['BsmtCond'])
mode_BsmtExposure_test = stat.mode(test_edited_object['BsmtExposure'])
mode_BsmtQual_test = stat.mode(test_edited_object['BsmtQual'])
mode_BsmtFinType1_test = stat.mode(test_edited_object['BsmtFinType1'])
mode_BsmtFinType2_test = stat.mode(test_edited_object['BsmtFinType2'])
mode_MasVnrType_test = stat.mode(test_edited_object['MasVnrType'])
mode_MSZoning_test = stat.mode(test_edited_object['MSZoning'])
mode_Functional_test = stat.mode(test_edited_object['Functional'])
mode_Utilities_test = stat.mode(test_edited_object['Utilities'])
mode_KitchenQual_test = stat.mode(test_edited_object['KitchenQual'])
mode_Exterior1st_test = stat.mode(test_edited_object['Exterior1st'])
mode_Exterior2nd_test = stat.mode(test_edited_object['Exterior2nd'])
mode_SaleType_test = stat.mode(test_edited_object['SaleType'])

test_edited_object['GarageCond'].fillna(mode_GarageCond_test, inplace=True)
test_edited_object['GarageQual'].fillna(mode_GarageQual_test, inplace=True)
test_edited_object['GarageFinish'].fillna(mode_GarageFinish_test, inplace=True)
test_edited_object['GarageType'].fillna(mode_GarageType_test, inplace=True)
test_edited_object['BsmtCond'].fillna(mode_BsmtCond_test, inplace=True)
test_edited_object['BsmtExposure'].fillna(mode_BsmtExposure_test, inplace=True)
test_edited_object['BsmtQual'].fillna(mode_BsmtQual_test, inplace=True)
test_edited_object['BsmtFinType2'].fillna(mode_BsmtFinType2_test, inplace=True)
test_edited_object['BsmtFinType1'].fillna(mode_BsmtFinType1_test, inplace=True)
test_edited_object['MasVnrType'].fillna(mode_MasVnrType_test, inplace=True)
test_edited_object['MSZoning'].fillna(mode_MSZoning_test, inplace=True)
test_edited_object['Functional'].fillna(mode_Functional_test, inplace=True)
test_edited_object['Utilities'].fillna(mode_Utilities_test, inplace=True)
test_edited_object['KitchenQual'].fillna(mode_KitchenQual_test, inplace=True)
test_edited_object['Exterior1st'].fillna(mode_Exterior1st_test, inplace=True)
test_edited_object['Exterior2nd'].fillna(mode_Exterior2nd_test, inplace=True)
test_edited_object['SaleType'].fillna(mode_SaleType_test, inplace=True)

# We replace the values of the categorical variables to be used in the classifier by numerical values.
train_edited_object['CentralAir'] = train_edited_object['CentralAir'].replace({'N': 0, 'Y': 1})
train_edited_object['PavedDrive'] = train_edited_object['PavedDrive'].replace({'N': 0, 'P': 1, 'Y': 2})

test_edited_object['CentralAir'] = test_edited_object['CentralAir'].replace({'N': 0, 'Y': 1})
test_edited_object['PavedDrive'] = test_edited_object['PavedDrive'].replace({'N': 0, 'P': 1, 'Y': 2})

train_edited_numeric['CentralAir'] = train_edited_object['CentralAir']
train_edited_numeric['PavedDrive'] = train_edited_object['PavedDrive']

test_edited_numeric['CentralAir'] = test_edited_object['CentralAir']
test_edited_numeric['PavedDrive'] = test_edited_object['PavedDrive']

# --------- Classifier ---------
feature_cols = ['OverallQual', 'TotalBsmtSF', 'FirstFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'CentralAir', 'PavedDrive']
X = train_edited_numeric[feature_cols]
y = train_edited_numeric.SalePrice

model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True, verbose=2,
                               max_features='sqrt')
model.fit(X, y)
y_pred = model.predict(test_edited_numeric[feature_cols])

output = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# 0.21058
