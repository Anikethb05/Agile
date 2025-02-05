#!/usr/bin/env python
# coding: utf-8

# In[1]:

print("Hello!")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("heart.csv")
df.head()


# In[3]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[4]:


numerical_cols=df.select_dtypes(include=['float64','int64']).columns


# In[5]:


numerical_cols


# In[6]:


df[numerical_cols]=scaler.fit_transform(df[numerical_cols])


# In[7]:


df


# In[8]:


X=df.drop('target',axis=1)
y=df['target']


# In[9]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.4,random_state=42)


# In[10]:


model=GaussianNB()


# In[11]:


model.fit(X_train,y_train)


# In[12]:


y_pred=model.predict(X_test)
y_pred


# In[13]:


#accuracy

print(model.score(X_test, y_test))


# In[14]:


accuracy_NB=accuracy_score(y_test,y_pred)
print(accuracy_NB)


# In[15]:


#Decision tree


# In[16]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


model1=DecisionTreeClassifier()


# In[19]:


model1.fit(X_train,y_train)


# In[20]:


y_pred=model1.predict(X_test)
y_pred


# In[21]:


accuracy_DT=accuracy_score(y_test,y_pred)
print(accuracy_DT)


# In[22]:


#Random Forest


# In[23]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


model2=RandomForestClassifier()


# In[26]:


model2.fit(X_train,y_train)


# In[27]:


y_pred=model2.predict(X_test)
y_pred


# In[28]:


accuracy_RF=accuracy_score(y_test,y_pred)
print(accuracy_DT)


# In[29]:


#SVM


# In[30]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[31]:


from sklearn.svm import SVC


# In[32]:


model3=SVC()


# In[33]:


model3.fit(X_train,y_train)


# In[34]:


y_pred=model3.predict(X_test)
y_pred


# In[35]:


accuracy_SVM=accuracy_score(y_test,y_pred)
print(accuracy_SVM)


# In[36]:


#KNN


# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


model4=KNeighborsClassifier(n_neighbors=3)


# In[39]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[40]:


model4.fit(X_train,y_train)


# In[41]:


y_pred=model4.predict(X_test)
y_pred


# In[42]:


accuracy_KNN=accuracy_score(y_test,y_pred)
print(accuracy_KNN)


# In[43]:


acc=[]
for i in range(1,50):
    m=KNeighborsClassifier(n_neighbors=i)
    m.fit(X_train,y_train)
    y_pred=m.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
plt.plot(acc)


# In[44]:


acc[3]


# In[45]:


model4=KNeighborsClassifier(n_neighbors=3)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model4.fit(X_train,y_train)
y_pred=model4.predict(X_test)
accuracy_KNN=accuracy_score(y_test,y_pred)


# In[46]:


#Logistic Resgression


# In[47]:


from sklearn.linear_model import LogisticRegression


# In[48]:


model5=LogisticRegression()


# In[49]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)


# In[50]:


model5.fit(X_train,y_train)


# In[51]:


y_pred=model5.predict(X_test)
y_pred


# In[52]:


accuracy_LR=accuracy_score(y_test,y_pred)
print(accuracy_LR)


# In[53]:


#plot


# In[54]:


y_plot=[accuracy_NB,accuracy_DT,accuracy_RF,accuracy_SVM,accuracy_KNN,accuracy_LR]
X_plot=['NB','DT','RF','SVM','KNN','LR']


# In[55]:


plt.plot(X_plot, y_plot, 'o', color='blue') 
plt.plot(X_plot, y_plot)  
plt.grid(True)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

