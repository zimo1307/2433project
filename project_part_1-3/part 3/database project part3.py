#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt 
from xgboost import XGBRFRegressor
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[4]:


data = pd.read_csv(r"C:\Users\13167\Desktop\term 1\database\project\archive\dataset.csv")


# In[5]:


#check null value
data.isnull().sum()


# In[6]:


#heatmap
plt.figure(figsize=(14,8))
sns.heatmap(data.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.title("Heatmap for correlation between columns")
plt.show()


# In[7]:


data.Age.describe()


# In[8]:


plt.figure(figsize=(10,5))
plt.hist(data.Age,edgecolor='k')
plt.xlabel("Age")
plt.ylabel("Count");
plt.title("Distribution of Age");


# In[9]:


sns.displot(data.Height)
plt.title("Distribution of height");


# In[10]:


#dependent and independent feature split
X = data.drop('PremiumPrice',axis=1)
y = data.PremiumPrice


# In[11]:


data.head()


# In[12]:


#normalization
scalar =  StandardScaler()
X.Age = scalar.fit_transform(X[['Age']])
X.Height = scalar.fit_transform(X[['Height']])
X.Weight = scalar.fit_transform(X[['Weight']])


# In[13]:


#train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=43)


# In[14]:


#model
models = {
    LinearRegression():'Linear Regression',
    Lasso():'Lasso',
    Ridge():'Ridge',
    XGBRFRegressor():'XGBRFRegressor',
    RandomForestRegressor():'RandomForest'
}
for m in models.keys():
    m.fit(X_train,y_train)


# In[15]:


for model,name in models.items():
     print(f"Accuracy Score for {name} is : ",model.score(X_test,y_test)*100,"%")


# In[16]:


#finding important feature 
#Linear Regression
linear =LinearRegression()
linear.fit(X_train,y_train)
feature_imp2 = linear.coef_
ax = sns.barplot(x=feature_imp2, y=X.columns)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[17]:


print('coef value:', linear.coef_)
print('intercept value:', linear.intercept_)


# In[18]:


coef_table = pd.DataFrame(list(X_train.columns)).copy()
coef_table.insert(len(coef_table.columns),"Coefs",linear.coef_.transpose())


# In[19]:


#random forest regressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train,y_train)
feature_imp1 = random_forest.feature_importances_
sns.barplot(x=feature_imp1, y=X.columns)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[20]:


#xgboost
xgboost =XGBRFRegressor()
xgboost.fit(X_train,y_train)
feature_imp2 = xgboost.feature_importances_
sns.barplot(x=feature_imp2, y=X.columns)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show();


# In[ ]:




