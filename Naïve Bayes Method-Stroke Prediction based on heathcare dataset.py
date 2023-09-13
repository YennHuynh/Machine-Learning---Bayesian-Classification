#!/usr/bin/env python
# coding: utf-8

# ### Reference Information

# This Jupyter Notebook provides an analysis of stroke prediction.

#  This dataset can be acquired from this link https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

# ### Overview

# #### Problem Domain

# In this Bayesian Classification analysis, the overarching objective is to gain insights into the likelihood of stroke occurrence in patients within the context of healthcare. Stroke is a significant global health concern, ranking as the second leading cause of death worldwide, responsible for approximately 11% of total fatalities according to the World Health Organization (WHO). 

# This dataset serves as a microcosm of the kind of information that can be gathered in attempts to better understand the trends and factors associated with stroke occurrences. By associating these attributes with individual health records, we are employing Bayesian Classification as a tool to delve deeper into this specific healthcare problem. This analysis aims to uncover potential connections between these patient attributes and the likelihood of experiencing a stroke, contributing to our broader understanding of stroke risk factors and prevention strategies.

# #### Objective

# Our primary objective is to understand the factors influencing stroke occurrence in patients based on a healthcare dataset. We will leverage Python programming along with libraries such as Pandas, NumPy, Matplotlib, and Seaborn to conduct a comprehensive exploration of the dataset. 

# We will conduct a thorough examination of the dataset, including the calculation of statistics, visualization of attribute distributions, and the identification of potential outliers. The Bayesian Classification modeling technique will play a pivotal role in addressing these questions. Specifically, we will use Bayesian Classification for binary classification, focusing on whether a patient is likely to experience a stroke or not.

# Ultimately, this analysis aims to shed light on the interplay between various patient characteristics and the probability of stroke occurrence, contributing to our understanding of stroke risk factors and providing insights for preventive healthcare measures. 

# ### Import Libraries and Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Setting initial Configuration

# In[2]:


pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 40)


# ### Data Ingestion & Initial Inspection

# In[3]:


#When importing 
df = pd.read_csv('/Users/yen/Desktop/healthcare-dataset-stroke-data.csv')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.dtypes


# ### Check for Missing Values

# In[10]:


df.isna().sum()


# ### Drop the ID  Field

# In[11]:


df = df.drop(['id'], axis=1)


# In[12]:


df


# ### Exploratory Data Analysis

# In[13]:


df.describe()


# In[14]:


#Summary statistics for all values
df.describe(include='all')


# In[15]:


df.describe(include='object')


# In[16]:


#Summary statistics with additional percentiles
df.describe(percentiles=[0.01,0.05,0.1,0.9,0.99,0.995])


# ### Corelation Analysis

# In[17]:


df.corr()


# ### Correlation between categorical fields and dependent variable based on p-value

# In[24]:


from scipy.stats import chi2_contingency


# In[25]:


# Assuming you have a DataFrame 'df' with your data
# 'CategoricalColumn' is the categorical predictor
# 'DependentVariable' is the categorical dependent variable

# Create a contingency table
contingency_table = pd.crosstab(df['gender'], df['stroke'])

# Calculate the Chi-Square statistic for the contingency table
chi2, p, _, _ = chi2_contingency(contingency_table)

# Create a heatmap of the Chi-Square statistic
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title(f'Chi-Square Heatmap (p-value = {p:.4f})')
plt.xlabel('stroke')
plt.ylabel('gender')
plt.show()


# In[26]:


contingency_table = pd.crosstab(df['ever_married'], df['stroke'])

# Calculate the Chi-Square statistic for the contingency table
chi2, p, _, _ = chi2_contingency(contingency_table)

# Create a heatmap of the Chi-Square statistic
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title(f'Chi-Square Heatmap (p-value = {p:.4f})')
plt.xlabel('stroke')
plt.ylabel('ever_married')
plt.show()


# In[27]:


contingency_table = pd.crosstab(df['work_type'], df['stroke'])

# Calculate the Chi-Square statistic for the contingency table
chi2, p, _, _ = chi2_contingency(contingency_table)

# Create a heatmap of the Chi-Square statistic
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title(f'Chi-Square Heatmap (p-value = {p:.4f})')
plt.xlabel('stroke')
plt.ylabel('work_type')
plt.show()


# In[28]:


contingency_table = pd.crosstab(df['Residence_type'], df['stroke'])

# Calculate the Chi-Square statistic for the contingency table
chi2, p, _, _ = chi2_contingency(contingency_table)

# Create a heatmap of the Chi-Square statistic
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title(f'Chi-Square Heatmap (p-value = {p:.4f})')
plt.xlabel('stroke')
plt.ylabel('Residence_type')
plt.show()


# In[29]:


contingency_table = pd.crosstab(df['smoking_status'], df['stroke'])

# Calculate the Chi-Square statistic for the contingency table
chi2, p, _, _ = chi2_contingency(contingency_table)

# Create a heatmap of the Chi-Square statistic
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title(f'Chi-Square Heatmap (p-value = {p:.4f})')
plt.xlabel('stroke')
plt.ylabel('smoking_status')
plt.show()


# If the p-value for a correlation coefficient is exactly 0 (p = 0), it typically indicates that there is an extremely strong and statistically significant correlation between the two variables being analyzed. In other words, a p-value of 0 suggests that the observed correlation in the data is highly unlikely to have occurred by random chance alone, and it provides strong evidence that there is a real and significant relationship between the variables.
# 
# 

# Null Hypothesis (H0): The null hypothesis, in this context, would typically be that there is no association or correlation between gender and the occurrence of strokes. It assumes that gender has no effect on stroke risk.

# P-Value = 0.7: A p-value of 0.7 is relatively large. It means that if the null hypothesis were true (i.e., if there were no true association between gender and stroke), there is a 70% chance of observing the data or data more extreme (in terms of the association between gender and stroke) purely by random variation.

# Interpretation: Since the p-value is high (0.7), you would typically fail to reject the null hypothesis. In practical terms, this suggests that, based on the available data and the statistical analysis conducted, there is insufficient evidence to conclude that there is a significant correlation between gender and the occurrence of strokes. The results do not support the idea that gender is a statistically significant predictor of stroke risk in this dataset.

# It's important to note that while a p-value of 0.7 suggests no statistical significance, the absence of a significant correlation in the dataset does not necessarily imply a lack of real-world association between gender and stroke risk. Other factors may be at play, and larger datasets or more refined statistical analyses may be needed to detect any potential relationships. Additionally, when interpreting p-values, it's essential to consider the significance level (alpha) and the specific context of the analysis.

# ### Plot Corelation for in a Heatmap

# In[30]:


sns.set(rc = {'figure.figsize':(15,8)})


# In[31]:


sns.heatmap(df.corr().abs(), annot = True,cmap = 'coolwarm')


# ### Data Visualization

# In[32]:


df.columns


# In[33]:


df.dtypes


# ### Categotical Variables

# In[34]:


df['gender'].value_counts()


# In[35]:


df['gender'].value_counts().plot(kind='bar',ylabel = 'Frequency', title='gender')
plt.show()


# In[36]:


# PIE CHART
#df['gender'].value_counts().plot(kind='pie',ylabel ="", title='gender', autopct = '%1.1f%',startangle=90)
#plt.show()


# In[37]:


df['smoking_status'].value_counts()


# In[38]:


df['ever_married'].value_counts().plot(kind='bar')
plt.show()


# In[39]:


df['work_type'].value_counts().plot(kind='bar')
plt.show()


# In[42]:


df['Residence_type'].value_counts().plot(kind='bar')
plt.show()


# In[43]:


df['smoking_status'].value_counts().plot(kind='bar')
plt.show()


# ### Numerical Values

# In[44]:


df['age'].plot(kind='hist')
plt.show()


# In[45]:


df.boxplot(column = 'age', vert =False)
plt.show()


# In[46]:


df['hypertension'].plot(kind='hist')
plt.show()


# In[47]:


df.boxplot(column = 'hypertension', vert =False)
plt.show()


# In[48]:


df['heart_disease'].plot(kind='hist')
plt.show()


# In[49]:


df.boxplot(column = 'heart_disease', vert =False)
plt.show()


# In[50]:


df['avg_glucose_level'].plot(kind='hist')
plt.show()


# In[51]:


df.boxplot(column = 'avg_glucose_level', vert =False)
plt.show()


# In[52]:


df['bmi'].plot(kind='hist')
plt.show()


# In[53]:


df.boxplot(column = 'bmi', vert =False)
plt.show()


# ### Data Preprocessing

# #### Replace Missing Values

# In[54]:


df.dtypes


# In[55]:


df.isna().sum()


# In[56]:


# Use mode for categorical variables
df.fillna(df.select_dtypes(include='object').mode().iloc[0], inplace = True) # start at row 0 column 0


# In[57]:


# Use mean for numerical variable or median if there are strong outliners
df.fillna(df.select_dtypes(include='number').mean().iloc[0], inplace = True)


# In[58]:


df.isna().sum()


# #### Handling Outliner

# In[59]:


# Define a function to replace outliers with the median
def handle_outliners(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(lower_bound)
    print(upper_bound)
    
    print("Original shape: ")
    print(df.shape)
    
    # Option 1: Remove outliers
    data = df[(series >= lower_bound) & (series <= upper_bound)]
    print("After remove outliners: ")
    print(data.shape)

    # Option 2: Replace outliers with the median
    #series = series.apply(lambda x: series.median() if x < lower_bound or x > upper_bound else x)
    #return series

# Apply the function to replace outliers with the median in the 'bmi' column
df['bmi'] = handle_outliners(df['bmi'])


# In[60]:


# Apply the function to replace outliers with the median in the 'avg_glucose_level' column
df['avg_glucose_level'] = handle_outliners(df['avg_glucose_level'])


# ### One Hot Encoding (for Input Variables)

# In[61]:


df.columns


# In[62]:


df.dtypes


# In[63]:


# dont endcode the target column 
cols = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']


# In[64]:


df = pd.get_dummies(df, columns = cols, drop_first=True)


# In[65]:


df


# ### Label Encoding (for target variable)

# In[66]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[67]:


df['stroke'] = le.fit_transform(df['stroke'])


# In[68]:


le.classes_


# In[69]:


df['stroke'].value_counts()


# In[70]:


df


# ### Shuffle the Dataset

# In[71]:


from sklearn.utils import shuffle


# In[72]:


df = shuffle(df)


# In[73]:


df


# ### Split into X and Y

# In[74]:


X = df.drop(['stroke'], axis = 1)
X


# In[75]:


y = df['stroke']
y


# ### Balance the Dataset

# In[76]:


import sklearn


# In[ ]:


#pip show scikit-learn


# In[ ]:


#print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[ ]:


# First uninstall the currently installed scikit-learn with the command: 


# In[ ]:


#pip uninstall scikit-learn --yes


# In[ ]:


# And then install the 0.22.2 version with the command: pip install scikit-learn==1.2.2


# In[ ]:


#pip install scikit-learn==1.2.2


# In[77]:


from collections import Counter


# In[78]:


from imblearn.over_sampling import RandomOverSampler


# In[79]:


# summarize class distribution
print(Counter(y))


# In[80]:


#define oversampling startegy
oversample = RandomOverSampler(sampling_strategy='minority')


# In[81]:


# fit and apply transform
X, y = oversample.fit_resample(X, y)


# In[82]:


# summarize class distribution
print(Counter(y))


# ### Normalize or Standardize the Dataset

# Standardizing a dataset involves rescaling the distribution of values so that the mean os observed values is 0 and the standard devistion is 1. This can be thought of as substracting the mean value or centering the data

# In[83]:


#from sklearn.preprocessing import StandardScaler


# In[84]:


#scaler_s= StandardScaler()


# In[85]:


#X= scaler_s.fit_transform(X.values)


# In[86]:


#X.shape


# Normalization Is a rescaling of the data from the original range so that all values are within the new range ot U and 1

# Normalization requires that you know or are able to accurately estimate the minimum and maximum observable values. You may be able to estimate these
# values from your available data.

# In[87]:


from sklearn.preprocessing import MinMaxScaler
scaler_m = MinMaxScaler()


# In[88]:


X = scaler_m.fit_transform(X.values)


# In[89]:


X.shape


# #### Train Test Split

# In[90]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size=.3)


# ### Modeling

# In[91]:


#import Bernoulli Naïve Bayes function from scikit-learn library
from sklearn.naive_bayes import BernoulliNB


# In[92]:


#initialize Bernoulli Naïve Bayes function to a variable
bnb = BernoulliNB()


# In[93]:


#build the model with training data
bnb.fit(X_train, y_train)


# In[94]:


#model's predictive score on the training data
bnb.score(X_train, y_train)


# In[95]:


#test the model on unseen data
#score predictive values in variable
y_pred = bnb.predict(X_test)


# ### Evaluation

# In[96]:


from sklearn.metrics import confusion_matrix, classification_report


# In[97]:


cm = confusion_matrix(y_test, y_pred)
cm


# ### Confusion Matrix

# In[98]:


import seaborn as sns
plt.figure(figsize= (10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# 980 instances were truly "Died" (True Died), and the model correctly predicted them as "Died" (Predicted Died). 477 instances were truly "Died" (True Died), but the model incorrectly predicted them as "Survived" (Predicted Survived). 187 instances were truly "Survived" (True Survived), but the model incorrectly predicted them as "Died" (Predicted Died). 1273 instances were truly "Survived" (True Survived), and the model correctly predicted them as "Survived" (Predicted Survived).

# In[99]:


#predictive score of the model on the test data
bnb.score(X_test, y_test)


# In[100]:


#predictive score of the model for each predictive category
print(classification_report(y_test, y_pred))


# In[ ]:




