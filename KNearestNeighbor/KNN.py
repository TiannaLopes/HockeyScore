#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[69]:


df = pd.read_csv('Boston.csv')
df = df.drop(columns='Date')
df = df.drop(columns='Type')
df = df.drop(columns='Opponent')
df = df.drop(columns='BruinsScore3')
df = df.drop(columns='OpponentScore3')
df = df.drop(columns='BruinsScoreF')
df = df.drop(columns='OpponentScoreF')
print(df.head())


# In[70]:


scaler = StandardScaler()
scaler.fit(df.drop('Result', axis=1))


# In[71]:


scaled_features = scaler.transform(df.drop('Result',axis=1))


# In[85]:


df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_feat)


# In[90]:


X = df_feat
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[91]:


knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)


# In[92]:


#Predict
pred = knn.predict(X_test)
print("Prediction")
print(pred)


# In[93]:


print(classification_report(y_test, pred))


# In[94]:


cmat = confusion_matrix(y_test, pred)
print('TP - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))


# In[95]:


error_rate = []
for i in range(1,50):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[96]:


plt.figure(figsize=(15,4))
plt.plot(range(1,50), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K-Values')
plt.xlabel('K-Values')
plt.ylabel('Error Rate')
plt.show()
