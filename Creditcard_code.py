#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df['Class'].value_counts()


# In[9]:


#Problem Statement
#This is a binary classification problem where we will build a model to try and predict if fraud occurred within transactions.

#The output label is Class:

#1 - Fraudulent
#0 - Genuine


# In[10]:


#check for missing data
df.isnull().sum()


# In[11]:


df["Class"].value_counts()


# In[12]:


#plotting correlation between the top features

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 20))


# In[13]:


#target label, features split
X = df.drop(["Class"], axis=1)
y = df.Class
y.head()


# In[14]:


X.head()


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(solver='liblinear')

log_model.fit(X_train, y_train)


# In[17]:


from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier()

forest_model.fit(X_train, y_train)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()

tree_model.fit(X_train, y_train)


# In[19]:


from sklearn.metrics import recall_score, precision_score, confusion_matrix

model_list = [tree_model, forest_model, log_model]

scores = {}

for model in model_list:
    y_preds = model.predict(X_test)
    score = recall_score(y_test, y_preds)
    scores[model] = score
    
for model, score in scores.items(): 
    print("{}".format(model) + " : " + "{}".format(score))    


# In[20]:


forest_model.get_params().keys()


# In[21]:


#Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

#tune the Random Forest Model
hyperparameter_grid = {
    'n_estimators' : [100, 150, 200],
'min_samples_split' : [3, 5, 6],
'max_leaf_nodes' : [5, 10, 15],
'max_features' : ["auto", "sqrt", "log2"]
}

# Set up the random search with cross validation
random_cv = RandomizedSearchCV(estimator=forest_model,
            param_distributions=hyperparameter_grid,
            cv=3, n_iter=20,
            scoring = 'recall',n_jobs = -1,
            verbose = 2, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train,y_train)


# In[22]:


random_cv.best_estimator_


# In[23]:


best_forest_model = RandomForestClassifier(max_features='sqrt', max_leaf_nodes=15,
                       min_samples_split=5, n_estimators=150)
best_forest_model.fit(X_train, y_train)

y_preds = best_forest_model.predict(X_test)
score = recall_score(y_test, y_preds)

print("Recall score: {}".format(score))


# In[24]:


#lets try the Decision Tree

#first we tune the Decision Tree Model
hyperparameter_grid = {
'min_samples_split' : [3, 5, 6],
'max_leaf_nodes' : [5, 10, 15],
'max_features' : ["auto", "sqrt", "log2"]
}

# Set up the random search with cross validation
random_cv = RandomizedSearchCV(estimator=tree_model,
            param_distributions=hyperparameter_grid,
            cv=3, n_iter=20,
            scoring = 'recall',n_jobs = -1,
            verbose = 2, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train,y_train)


# In[25]:


random_cv.best_estimator_


# In[26]:


best_tree_model = DecisionTreeClassifier(max_features='sqrt', max_leaf_nodes=15,
                       min_samples_split=5)

best_tree_model.fit(X_train, y_train)
y_preds = best_tree_model.predict(X_test)
score = recall_score(y_test, y_preds)

print("Recall score: {}".format(score))


# In[27]:


forest_model.fit(X, y)


# In[ ]:




