
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv('USA_Housing.csv')


# In[10]:


df.head()


# In[11]:


df.describe()


# In[12]:


df.columns


# In[11]:


sns.pairplot(df)


# In[13]:


sns.distplot(df['Price'])


# In[21]:


sns.heatmap(df.corr(), annot=True)


# In[21]:


df.columns


# In[31]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']


# In[32]:


y=df['Price']


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


lm=LinearRegression()


# In[37]:


lm.fit(X_train,y_train)


# In[38]:


print(lm.intercept_)


# In[40]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[48]:


predictions = lm.predict(X_test)


# In[49]:


plt.scatter(y_test,predictions)


# In[50]:


sns.distplot((y_test-predictions),bins=50);


# In[51]:


from sklearn import metrics


# In[53]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

