#!/usr/bin/env python
# coding: utf-8

#                             Data Visualization and Pre-processing

#                                       Assigment -2

# 1. Download the dataset Dataset

# My dataset is downloaded in the path: C:\Users\STUDENT\Downloads\Churn_Modelling

# 2.Load the dataset

# In[5]:


#import library
import pandas as pd


# In[6]:


dataset = pd.read_csv(r"C:\Users\STUDENT\Downloads\Churn_Modelling.csv")
dataset.head()


# 3.Perform Below Visualization ● Univariate Analysis● Bi - Variate Analysis● Multi - Variate Analysis

# Unvariate Analysis

# In[9]:


#IMPORT SEABORN LIBRARY
import seaborn as sns


# In[10]:


sns.displot(dataset['CreditScore'])


# Bi-Variate Analysis:-

# In[15]:


sns.relplot(x="CreditScore",y='Age',data=dataset)


# In[16]:


sns.relplot(x="CreditScore",y='Age',hue="IsActiveMember",data=dataset)


# 4.. Perform descriptive statistics on the dataset

# In[17]:


import pandas as pd
import numpy as np
ds = pd.read_csv(r"C:\Users\STUDENT\Downloads\Churn_Modelling.csv")
ds.head(2)


# In[18]:


ds.isnull().any()


# In[19]:


#to get all statiscal values
ds.describe()


# 5. Handle the Missing values

# In[20]:


dataset.head()


# In[21]:


dataset.isnull().sum()


# Thus,The dataset is not having my missing or null values.
#  If an dataset will have any missing values,we can handle it in following ways
#  1) Lot of misssing values---remove
#  2) Less missing values---replace
# Function used---fillna()

# In[22]:


#so, we no need to handle missing values in this dataset


# 6. Find the outliers and replace the outliers

# In[23]:


#finding the outlier
dataset.skew()


# In[26]:


sns.boxplot(dataset["Age"])


# In[28]:


q1= dataset["Age"].describe()["25%"]
q3= dataset["Age"].describe()["75%"]


# In[29]:


q1


# In[30]:


q3


# In[31]:


iqr=q3-q1
iqr


# In[37]:


l_b=q1-(1.5*iqr)
u_b=q3+(1.5*iqr)


# In[38]:


l_b


# In[39]:


l_b=q1-(1.5*iqr)
u_b=q3+(1.5*iqr)


# In[40]:


l_b


# In[41]:


u_b


# In[42]:


dataset[dataset["Age"]<l_b]


# In[43]:


dataset[dataset["Age"]<l_b]


# In[44]:


dataset[dataset["Age"]>u_b].head()


# In[45]:


#replace the outlier
dataset.dtypes


# In[46]:


outlier_list=list(dataset[dataset["Age"]>u_b]["Age"])
outlier_list


# In[47]:


outlier_dict={}.fromkeys(outlier_list,u_b)
outlier_dict


# In[48]:


dataset["Age"]=dataset["Age"].replace(outlier_dict)
sns.boxplot(dataset["Age"])


# 7. Check for Categorical columns and perform encoding
# 8. Split the data into dependent and independent variables
# 9. Scale the independent variables

# In[49]:


dataset.isnull().any()


# In[51]:


dataset["CustomerId"].unique()


# In[52]:


dataset["CustomerId"].unique()


# In[56]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
dataset.head()


# In[74]:


x=dataset.iloc[:,0:4].values
x


# In[75]:


type(x)


# In[77]:


y=dataset.iloc[:,4:5].values


# In[78]:


x.shape


# In[79]:


y.shape


# In[80]:


ct=ColumnTransformer([("oh",OneHotEncoder(),[3])],remainder="passthrough")


# In[81]:


#x = ct.fit_transform(x)
x


# 10. Split the data into training and testing

# In[83]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[84]:


x_train.shape


# In[85]:


x_test.shape


# In[86]:


y_train.shape


# In[87]:


y_test.shape


# In[ ]:




