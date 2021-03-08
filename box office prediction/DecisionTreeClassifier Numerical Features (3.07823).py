# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pandas.core.common import SettingWithCopyWarning
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/)
# that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')
train.head()

test = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')
test.head()

train.describe()

train['revenue'].describe()

plt.hist(train['revenue'])

test.describe()

train.info()

test.info()

train.isnull().sum().sort_values(ascending=False)

test.isnull().sum().sort_values(ascending=False)

# belongs_to_collection (object), homepage (object) y tagline (object) have too many null values in both datasets.
# Therefore, we eliminate them.
# The id, poster-path, status and overview features either, they do not provide information.

features_to_drop = ['belongs_to_collection', 'homepage', 'tagline', 'id',
                    'imdb_id', 'overview', 'poster_path', 'status']
train_edited = train

for element in features_to_drop:
    train_edited = train_edited.drop([element], axis=1)

train_edited.head()

test_edited = test

for element in features_to_drop:
    test_edited = test_edited.drop([element], axis=1)

test_edited.head()

# Let's make a general preliminary analysis

# 10 highest-grossing movies
train_edited[['title', 'release_date', 'budget', 'revenue']].sort_values(['revenue'], ascending=False)\
    .head(10).style.background_gradient(subset='revenue', cmap='BuGn')

# Now, we have to compare the features with the final tag. First of all, we're going to compare the numerical fields
train_edited_numeric = train_edited.select_dtypes(np.number)

train_edited_numeric.isnull().sum().sort_values(ascending=False)

# We calculate the median for the field and then replace the null values by these value
median_runtime = train_edited_numeric['runtime'].median()

# Ignore some warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

train_edited_numeric['runtime'].fillna(median_runtime, inplace=True)

train_edited_numeric.isnull().sum().sort_values(ascending=False)

# We now proceed to plot each of the above variables respect to the final price using scatter diagrams
# in order to see any relationship between them

revenue = train_edited_numeric['revenue']

numeric_columns = train_edited_numeric.columns
numeric_columns = numeric_columns[:len(numeric_columns)]

fig = plt.figure()
ncols = len(numeric_columns)
order = 0
fig, axes = plt.subplots(ncols=ncols, sharey=True, figsize=(18, 5))
for column in range(0, ncols):
    formula = 'revenue' + '~' + str(numeric_columns[order])
    axes[column].scatter(train_edited_numeric[numeric_columns[order]], revenue)
    axes[column].set_xlabel(numeric_columns[order], labelpad=5)
    lm = smf.ols(formula=formula, data=train_edited_numeric).fit()
    expected_price = lm.predict(pd.DataFrame(train_edited_numeric[numeric_columns[order]]))
    axes[column].plot(pd.DataFrame(train_edited_numeric[numeric_columns[order]]), expected_price, c="red", linewidth=2)
    order = order + 1


plt.figure(figsize=(10, 5))
correlation_mat = train_edited_numeric.corr()
sns.heatmap(correlation_mat, annot=True)
plt.show()


corr_pairs = correlation_mat.unstack()
print(corr_pairs['revenue'])

test_edited_numeric = test_edited.select_dtypes(np.number)

test_edited_numeric.isnull().sum().sort_values(ascending=False)

median_runtime_test = test_edited_numeric['runtime'].median()

test_edited_numeric['runtime'].fillna(median_runtime_test, inplace=True)

# --------- Classifier ---------
feature_cols = ['budget', 'popularity']

# X = Features
X = train_edited_numeric[feature_cols]
# y = target variable
y = train_edited_numeric.revenue

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X, y)

# Predict the response for test dataset
y_pred = clf.predict(test_edited_numeric[feature_cols])

output = pd.DataFrame({'id': test.id, 'revenue': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# 3.07823
