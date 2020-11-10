#!/usr/bin/env python
# coding: utf-8

# #                           THE SPARKS FOUNDATION
#                                          (GRIPNOV20)
# Author : BEMBERKAR SHASHANK SAI
# 
#     DATASCIENCE AND BUSINESS ANALYTICS
#     
#     TASK 1: Predict the percentage of an student based on the number of study hours using linear regression
# 
#     

# In[1]:


# IMPORTING REQUIRED LIBRARIES FOR THE DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='white',color_codes=True)
sns.set(font_scale=1.5)


# In[2]:


#IMPORTING DATA
df=pd.read_csv('data.csv')
df


# In[3]:


#EXPLORATORY DATA ANALYSIS
df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum() #for null values


# In[10]:


df.isna().any  #for missing values


# In[11]:


#plotting the distribution 
df.plot.scatter(x='Hours',y='Scores',color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[12]:


sns.heatmap(df.corr(),annot=True)


# In[13]:


df.corr()


# In[14]:


sns.lmplot(x="Hours",y="Scores",data=df)
plt.title('plotting the regreesion line')


# In[15]:


#PREPARING THE DATA
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[17]:


#TRAINING THE MODEL
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state = 0)
linreg = LinearRegression()
linreg.fit(x,y)
print("done with the training")


# In[18]:


#VISUALIZING THE REGRESSION LINE
line = linreg.coef_*x + linreg.intercept_
plt.scatter(x,y,color ='blue')
plt.plot(x,line,color ='red')
plt.show()


# In[19]:


#PREDICTING TRAINING SAMPLES
Y_pred = linreg.predict(X_train)

print("\nTraining score :")
print("Mean squared error: %.2f"% mean_squared_error(Y_train, Y_pred))
print('R2 score: %2f' % r2_score(Y_train, Y_pred))

#PREDICTING TESTING SAMPLES

Y_pred = linreg.predict(X_test)


print("\nTesting score :")
print("Mean squared error: %.2f"% mean_squared_error(Y_test, Y_pred))
print('R2 score: %2f' % r2_score(Y_test, Y_pred))


# In[20]:


#TESTING ON OUR OWN DATA
Hours = np.array([[9.25]])
predict=linreg.predict(Hours)
print("No of Hours = {}".format(Hours))
print("predicted score = {}".format(predict[0]))


# # THANKYOU
