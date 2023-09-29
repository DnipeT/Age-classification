import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression # LogisticRegression also works pretty good

InputPath = '' # testing data's path

df1 = pd.read_csv('') # training data's path for profile
df2 = pd.read_csv('') # training data's path for relation

df2['like_id'] = df2['like_id'].astype(str)
df3 = pd.merge(df1, df2, on='userid')

df4 = pd.DataFrame(df3.groupby(['userid', 'age'])['like_id'].apply(list).reset_index(name='like_id')) 


# convert list of (list of string) into list of string for CountVectorizer() to work
index = 0
for i in df4['like_id']:
    df4.at[index, 'like_id'] = ' '.join(i)
    index = index + 1

data_YouTube = df4.loc[:, ['userid','age', 'like_id']]

ageGroupList = []
for i in data_YouTube.index:
    if data_YouTube['age'].iloc[i] <= 24:
        ageGroupList.append("xx-24")
    elif data_YouTube['age'].iloc[i] <= 34:
        ageGroupList.append("25-34")
    elif data_YouTube['age'].iloc[i] <= 49:
        ageGroupList.append("35-49")
    else:
        ageGroupList.append("50-xx")

data_YouTube['AgeGroup'] = ageGroupList

print(data_YouTube[['AgeGroup', 'age']])

# Splitting the data into 1000 training instances and 8500 test instances
n = 1000
all_Ids = np.arange(len(data_YouTube))
# random.shuffle(all_Ids). it does not really help me at all.
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = data_YouTube.loc[test_Ids, :]
data_train = data_YouTube.loc[train_Ids, :]


# Training a Naive Bayes model
count_vect = CountVectorizer()


X_train = count_vect.fit_transform(data_train['like_id'])
y_train = data_train['AgeGroup']
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Testing the Naive Bayes model

X_test = count_vect.transform(data_test['like_id'])
y_test = data_test['AgeGroup']
y_predicted = clf.predict(X_test)


# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test, y_predicted))

