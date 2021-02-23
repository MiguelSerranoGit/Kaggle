# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a
# version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# We see the train data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

# We see the test data
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

# We represent the train data with histograms to get a first approximation of the distribution of the data
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
axs[0, 0].hist(train_data.Survived)
axs[0, 0].set_title('Survived')
axs[0, 1].hist(train_data.Pclass)
axs[0, 1].set_title('Pclass')
axs[0, 2].hist(train_data.Sex)
axs[0, 2].set_title('Sex')
axs[0, 3].hist(train_data.Age)
axs[0, 3].set_title('Age')
axs[1, 0].hist(train_data.SibSp)
axs[1, 0].set_title('SibSp')
axs[1, 1].hist(train_data.Parch)
axs[1, 1].set_title('Parch')
axs[1, 2].hist(train_data.Fare)
axs[1, 2].set_title('Fare')
# axs[1,3].hist(train_data.Embarked)

# Training set statistics and information
train_data.describe()
train_data.info()
train_data.isnull().sum().sort_values(ascending=False)
# Missing data in Cabin (687), Age (177) and Embarked (2) columns

# Drop the Cabin column
train_data = train_data.drop(['Cabin'], axis=1)
# Subsequently, the rest of the null values must be substituted.


# We analyze each passenger feature with the exit tag:
# - Pclass
train_grouped_by_class_survived = train_data.groupby(['Pclass', 'Survived']).count()
train_grouped_by_class_survived['PassengerId'].plot.bar()
plt.grid()
# Class	Died	Survived
# 1st	37.04%	62.96%
# 2nd	52.71%	47.29%
# 3rd	75.76%	24.24%
# It is very likely that the third class passengers died

# - Sex
train_grouped_by_sex_survived = train_data.groupby(['Sex', 'Survived']).count()

train_grouped_by_sex_survived['PassengerId'].plot.bar()
plt.grid()
# Sex	Died	Survived
# Male	81.11%	18.89%
# Female	25.79%	74.21%
# Men are more likely to die and women are more likely to survive

# - Age
# Null values must be substituted
train_data['Age'].isnull().sum()
# 177
test_data['Age'].isnull().sum()
# 86

train_grouped_by_sex_pclass = train_data.groupby(['Sex', 'Pclass']).mean()
test_grouped_by_sex_pclass = test_data.groupby(['Sex', 'Pclass']).mean()

# We calculate the average age of each sex and each class of both datasets to be more accurate.
mean_age_female_1class = round(train_grouped_by_sex_pclass['Age'].take([0]), 2)
mean_age_female_2class = round(train_grouped_by_sex_pclass['Age'].take([1]), 2)
mean_age_female_3class = round(train_grouped_by_sex_pclass['Age'].take([2]), 2)
mean_age_male_1class = round(train_grouped_by_sex_pclass['Age'].take([3]), 2)
mean_age_male_2class = round(train_grouped_by_sex_pclass['Age'].take([4]), 2)
mean_age_male_3class = round(train_grouped_by_sex_pclass['Age'].take([5]), 2)

mean_age_female_1class_test = round(test_grouped_by_sex_pclass['Age'].take([0]), 2)
mean_age_female_2class_test = round(test_grouped_by_sex_pclass['Age'].take([1]), 2)
mean_age_female_3class_test = round(test_grouped_by_sex_pclass['Age'].take([2]), 2)
mean_age_male_1class_test = round(test_grouped_by_sex_pclass['Age'].take([3]), 2)
mean_age_male_2class_test = round(test_grouped_by_sex_pclass['Age'].take([4]), 2)
mean_age_male_3class_test = round(test_grouped_by_sex_pclass['Age'].take([5]), 2)


# Male
# Train
is_male = train_data.loc[:, 'Sex'] == 'male'
train_data_male = train_data.loc[is_male]

# 1st class
is_first_class_male = train_data_male.loc[:, 'Pclass'] == 1
train_data_male_1class = train_data_male.loc[is_first_class_male]
train_data_male_1class['Age'].fillna(float(mean_age_male_1class), inplace=True)

# 2nd class
is_second_class_male = train_data_male.loc[:, 'Pclass'] == 2
train_data_male_2class = train_data_male.loc[is_second_class_male]
train_data_male_2class['Age'].fillna(float(mean_age_male_2class), inplace=True)

# 3rd class
is_third_class_male = train_data_male.loc[:, 'Pclass'] == 3
train_data_male_3class = train_data_male.loc[is_third_class_male]
train_data_male_3class['Age'].fillna(float(mean_age_male_3class), inplace=True)

# Test
is_male = test_data.loc[:, 'Sex'] == 'male'
test_data_male = test_data.loc[is_male]

# 1st class
is_first_class_male = test_data_male.loc[:, 'Pclass'] == 1
test_data_male_1class = test_data_male.loc[is_first_class_male]
test_data_male_1class['Age'].fillna(float(mean_age_male_1class_test), inplace=True)

# 2nd class
is_second_class_male = test_data_male.loc[:, 'Pclass'] == 2
test_data_male_2class = test_data_male.loc[is_second_class_male]
test_data_male_2class['Age'].fillna(float(mean_age_male_2class_test), inplace=True)

# 3rd class
is_third_class_male = test_data_male.loc[:, 'Pclass'] == 3
test_data_male_3class = test_data_male.loc[is_third_class_male]
test_data_male_3class['Age'].fillna(float(mean_age_male_3class_test), inplace=True)

# Female
# Train
is_female = train_data.loc[:, 'Sex'] == 'female'
train_data_female = train_data.loc[is_female]

# 1st class
is_first_class_female = train_data_female.loc[:, 'Pclass'] == 1
train_data_female_1class = train_data_female.loc[is_first_class_female]
train_data_female_1class['Age'].fillna(float(mean_age_female_1class), inplace=True)

# 2nd class
is_second_class_female = train_data_female.loc[:, 'Pclass'] == 2
train_data_female_2class = train_data_female.loc[is_second_class_female]
train_data_female_2class['Age'].fillna(float(mean_age_female_2class), inplace=True)

# 3rd class
is_third_class_female = train_data_female.loc[:, 'Pclass'] == 3
train_data_female_3class = train_data_female.loc[is_third_class_female]
train_data_female_3class['Age'].fillna(float(mean_age_female_3class), inplace=True)

# Test
is_female = test_data.loc[:, 'Sex'] == 'female'
test_data_female = test_data.loc[is_female]

# 1st class
is_first_class_female = test_data_female.loc[:, 'Pclass'] == 1
test_data_female_1class = test_data_female.loc[is_first_class_female]
test_data_female_1class['Age'].fillna(float(mean_age_female_1class_test), inplace=True)

# 2nd class
is_second_class_female = test_data_female.loc[:, 'Pclass'] == 2
test_data_female_2class = test_data_female.loc[is_second_class_female]
test_data_female_2class['Age'].fillna(float(mean_age_female_2class_test), inplace=True)

# 3rd class
is_third_class_female = test_data_female.loc[:, 'Pclass'] == 3
test_data_female_3class = test_data_female.loc[is_third_class_female]
test_data_female_3class['Age'].fillna(float(mean_age_female_3class_test), inplace=True)


# We have separated each dataframe and substituted the corresponding values. Now we must put them back together
dataframes = [train_data_male_2class, train_data_male_3class, train_data_female_1class, train_data_female_2class,
              train_data_female_3class]
train_data_without_ages_null = train_data_male_1class.append(dataframes, sort=False)
train_data_without_ages_null = train_data_without_ages_null.sort_values('PassengerId')


dataframes = [test_data_male_2class, test_data_male_3class, test_data_female_1class, test_data_female_2class,
              test_data_female_3class]
test_data_without_ages_null = test_data_male_1class.append(dataframes, sort=False)
test_data_without_ages_null = test_data_without_ages_null.sort_values('PassengerId')


plt.hist(train_data_without_ages_null['Age'])

plt.hist(test_data_without_ages_null['Age'])

# Histogram of ages as a function of survivors and deceased.
survived = train_data_without_ages_null.loc[:, 'Survived'] == 1
train_data_without_ages_null_survived = train_data_without_ages_null.loc[survived]

died = train_data_without_ages_null.loc[:, 'Survived'] == 0
train_data_without_ages_null_died = train_data_without_ages_null.loc[died]

fig1, axs = plt.subplots(1, 2, sharey=True)
axs[0].hist(train_data_without_ages_null_survived['Age'])
axs[0].set_title('Survived')
axs[1].hist(train_data_without_ages_null_died['Age'])
axs[1].set_title('Died')


# - Parch
train_grouped_by_parch_survived = train_data.groupby(['Parch', 'Survived']).count()

train_grouped_by_parch_survived['PassengerId'].plot.bar()
plt.grid()

# Parch	Died Survived
# 0	65.63%	34.37%
# 1	44.91%	55.09%
# 2	50%	    50%
# 3	40% 	60%
# 4	100%	0%
# 5	80%	    20%
# 6	100%	0%

# - Fare
fig2, axs = plt.subplots(1, 2, sharey=True)
axs[0].hist(train_data_without_ages_null_survived['Fare'])
axs[0].set_title('Survived')
axs[1].hist(train_data_without_ages_null_died['Fare'])
axs[1].set_title('Died')

# - Embarked
train_grouped_by_embarked_survived = train_data.groupby(['Embarked', 'Survived']).count()

train_grouped_by_embarked_survived['PassengerId'].plot.bar()
plt.grid()

# - SibSp
train_grouped_by_SibSp_survived = train_data.groupby(['SibSp', 'Survived']).count()

train_grouped_by_SibSp_survived['PassengerId'].plot.bar()
plt.grid()

# We check again if there are null values in the dataset
train_data_without_ages_null.isnull().sum().sort_values(ascending=False)
# Embarked (2)
test_data_without_ages_null.isnull().sum().sort_values(ascending=False)
# Cabin (327) Fare (1)

# We calculate the mean of the Fare values of the test set.
mean_fare = test_data_without_ages_null.Fare.mean()
test_data_without_ages_null['Fare'].fillna(float(mean_fare), inplace=True)

# As the decision tree we are going to use cannot have categorical variables, we replace the values by 0 and 1.
train_data_without_ages_null = train_data_without_ages_null.replace({'male': 0, 'female': 1})
test_data_without_ages_null = test_data_without_ages_null.replace({'male': 0, 'female': 1})

# --------- Classifier ---------
feature_cols = ['Pclass', 'Sex']
# X = Features
X = train_data_without_ages_null[feature_cols]
# y = target variable
y = train_data_without_ages_null.Survived

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X, y)

# Predict the response for test dataset
y_pred = clf.predict(test_data_without_ages_null[feature_cols])

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# Result: 0.77511
