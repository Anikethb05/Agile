#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("lungcancer.csv")
df.head()


# In[3]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['GENDER']=encoder.fit_transform(df[['GENDER']])


# In[4]:


df['LUNG_CANCER']=encoder.fit_transform(df[['LUNG_CANCER']])


# In[5]:


df.head()


# In[6]:


X=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']


# In[7]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[8]:


model=GaussianNB()


# In[9]:


model.fit(X_train,y_train)


# In[10]:


y_pred=model.predict(X_test)
y_pred


# In[11]:


#accuracy

print(model.score(X_test, y_test))


# In[12]:


accuracy_NB=accuracy_score(y_test,y_pred)
print(accuracy_NB)


# In[13]:


#Decision tree


# In[14]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[15]:


from sklearn.tree import DecisionTreeClassifier


# In[16]:


model1=DecisionTreeClassifier()


# In[17]:


model1.fit(X_train,y_train)


# In[18]:


y_pred=model1.predict(X_test)
y_pred


# In[19]:


accuracy_DT=accuracy_score(y_test,y_pred)
print(accuracy_DT)


# In[20]:


#Random Forest


# In[21]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)


# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


model2=RandomForestClassifier()


# In[24]:


model2.fit(X_train,y_train)


# In[25]:


y_pred=model2.predict(X_test)
y_pred


# In[26]:


accuracy_RF=accuracy_score(y_test,y_pred)
print(accuracy_DT)


# In[27]:


#SVM


# In[28]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[29]:


from sklearn.svm import SVC


# In[30]:


model3=SVC()


# In[31]:


model3.fit(X_train,y_train)


# In[32]:


y_pred=model3.predict(X_test)
y_pred


# In[33]:


accuracy_SVM=accuracy_score(y_test,y_pred)
print(accuracy_SVM)


# In[34]:


#KNN


# In[35]:


from sklearn.neighbors import KNeighborsClassifier


# In[36]:


model4=KNeighborsClassifier(n_neighbors=3)


# In[37]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[38]:


model4.fit(X_train,y_train)


# In[39]:


y_pred=model4.predict(X_test)
y_pred


# In[40]:


accuracy_KNN=accuracy_score(y_test,y_pred)
print(accuracy_KNN)


# In[41]:


acc=[]
for i in range(1,50):
    m=KNeighborsClassifier(n_neighbors=i)
    m.fit(X_train,y_train)
    y_pred=m.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
plt.plot(acc)


# In[42]:


acc[5]


# In[43]:


model4=KNeighborsClassifier(n_neighbors=5)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model4.fit(X_train,y_train)
y_pred=model4.predict(X_test)
accuracy_KNN=accuracy_score(y_test,y_pred)


# In[44]:


#Logistic Resgression


# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


model5=LogisticRegression()


# In[47]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)


# In[48]:


model5.fit(X_train,y_train)


# In[49]:


y_pred=model5.predict(X_test)
y_pred


# In[50]:


accuracy_LR=accuracy_score(y_test,y_pred)
print(accuracy_LR)


# In[51]:


#plot


# In[52]:


y_plot=[accuracy_NB,accuracy_DT,accuracy_RF,accuracy_SVM,accuracy_KNN,accuracy_LR]
X_plot=['NB','DT','RF','SVM','KNN','LR']


# In[53]:


plt.plot(X_plot, y_plot, 'o', color='blue') 
plt.plot(X_plot, y_plot)  
plt.grid(True)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

