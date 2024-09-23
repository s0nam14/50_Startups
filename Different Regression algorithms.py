#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Checking the version of Python packages (libraries)
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
print('Python: {}'.format(sys.version),"\n",
     'Scipy: {}'.format(scipy.__version__),"\n",
     'numpy: {}'.format(numpy.__version__),"\n",
     'matplotlib: {}'.format(matplotlib.__version__),"\n",
     'pandas: {}'.format(pandas.__version__),"\n",
     'sklearn: {}'.format(sklearn.__version__))


# In[3]:


## Load the needed Libraries
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


# In[4]:


df = pd.read_csv("50_startups.csv")


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


## Data Visualization
import seaborn as sns
sns.pairplot(df)


# In[8]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm');


# In[9]:


## Independent and Dependent features
X = df[['R&D Spend','Administration','Marketing Spend']]
y = df[['Profit']]


# In[10]:


## Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state=42)


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[12]:


## Python Function
def eval_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    score = r2_score(actual, pred)
    return print ("mae:", mae, "\n",
                 "mse:", mse, "\n",
                 "rmse:", rmse, "\n",
                 "r2_score:", score, "\n")


# In[13]:


## 1. Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
eval_metrics (y_test, y_pred)


# In[14]:


## 2. Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
eval_metrics (y_test, y_pred)


# In[15]:


## 3. Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
eval_metrics (y_test, y_pred)


# In[16]:


## 4. Elastic Net
en_model = ElasticNet()
en_model.fit(X_train, y_train)
y_pred = en_model.predict(X_test)
eval_metrics (y_test, y_pred)


# In[17]:


## 5. KNeighborsRegressor
KNR_model = KNeighborsRegressor()
KNR_model.fit(X_train, y_train)
y_pred = KNR_model.predict(X_test)
eval_metrics (y_test, y_pred)


# In[18]:


## 6. DecisionTreeRegressor
DTR_model = DecisionTreeRegressor()
DTR_model.fit(X_train, y_train)
y_pred = DTR_model.predict(X_test)
eval_metrics (y_test, y_pred)


# In[19]:


## 7. Random Forest Regressor
RFR_model = RandomForestRegressor()
RFR_model.fit(X_train, y_train)
y_pred = RFR_model.predict(X_test)
eval_metrics (y_test, y_pred)


# In[ ]:




