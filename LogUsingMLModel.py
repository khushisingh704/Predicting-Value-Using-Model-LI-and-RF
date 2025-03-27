#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('data.csv')
df


# # Data Perparation

# ## Data seperation as x and y

# In[3]:


df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) #Removed spaces from all table using function lambda
df


# In[4]:


y = df['logS']
y


# In[5]:


x = df.drop('logS', axis=1)
x


# # Splitting Training and Test data

# In[6]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2, random_state=100) # we basically gave test data to be 20% of real data

#basically train_test_split function perfroming trial on 80% and testing on 20% of the real data in random way and storing it in 
  # var x_train=training data of x stored automatically by train_test_split() x_test = testing data of x stored automatically by same function, same goes for y


# In[7]:


x_train


# In[8]:


x_test


# # Model Building

# ## Linear Building

# In[9]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train) # we built the linear regression model on training data


# # Applying the model to make prediction

# In[10]:


y_lr_train_pred = lr.predict(x_train) # predicting data for y(diff y_train as that was already in table but here we are predicting y value(log) in training data) using x_train data
y_lr_test_pred = lr.predict(x_test) # predicting data for y (same above line just its for test data)using x_test data


# In[11]:


y_lr_test_pred


# # Evaluate Model Performance

# In[12]:


from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred) #to check whether model is predicting right value or not compared to training values
lr_test_r2 = r2_score(y_test, y_lr_test_pred) # updated formula to check prediction is right or not pf test data compared to training data


# In[13]:


lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()

lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'] 

lr_results 


# # Random Forest

# ## Training the model

# In[14]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)

rf.fit(x_train, y_train)


# ## Applying the model to predict

# In[15]:


y_rf_train_pred = rf.predict(x_train) # predicting data for y(diff y_train as that was already in table but here we are predicting y value(log) in training data) using x_train data
y_rf_test_pred = rf.predict(x_test)


# ## Evaluate model perfromance

# In[16]:


from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred) #to check whether model is predicting right value or not compared to training values
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


# In[17]:


rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()

rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'] 

rf_results 


# ## Model Comparison

# In[22]:


df_models = pd.concat([lr_results,rf_results],axis=0)
df_models


# In[23]:


df_models.reset_index(drop=True) # Formatted index value 0 and 1


# # Data VisualIzation

# In[33]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y = y_lr_train_pred,alpha=0.3) 

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train,p(y_train),'#F8766D')

plt.ylabel('Train Predict LogS')
plt.xlabel('Train Actual LogS')


# In[ ]:




